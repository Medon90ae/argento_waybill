# main.py
# FastAPI service for address extraction (AraBERT + CAMeL optional) + dictionary/fuzzy fallback
# Endpoint: POST /extract  { "address": "..." }
# Returns: { city, area, clean_address, score, matched_on, suggestions: [...] }

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import re, json, os, logging

# Optional ML imports (attempted; if unavailable, we fallback to rule-based)
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

try:
    # camel_tools is optional but recommended
    from camel_tools.utils.normalize import normalize_unicode, normalize_alef_maksura_ar
    CAMEL_TOOLS_AVAILABLE = True
except Exception:
    CAMEL_TOOLS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("address-extractor")

app = FastAPI(title="Address Extractor API (Speedaf PoC)")

class ExtractRequest(BaseModel):
    address: str
    use_fuzzy: Optional[bool] = True

class ExtractResponse(BaseModel):
    city: Optional[str]
    area: Optional[str]
    clean_address: str
    score: float
    matched_on: Optional[str]
    suggestions: Optional[List[dict]] = []

# ----------------- dictionary (your provided cities) -----------------
CITIES_DICT = {
    "Cairo": ["Abdin","El Abaseya","Al Azbakeya","El Katameya","Al Manial","El Mokattam","Dar El Salam","El Basatin","El-Gamaleya","Al Sayeda Zeinab","Helwan","Maadi","Masr El Qadima","Qasr El Nil","El Khalifa","Torah","Zamalek","El Shorouk","Badr City","Ain Shams","Madinaty","Al Zaitoun","Nasr City","New Cairo","Heliopolis","Al Amiriyyah","El Salam","El Waily","Al Sharabeya","Al Marj","Al Matariyyah","El Nozha","Hadaeq Al Qubbah","Shobra","Nile Corniche","Cairo"],
    "Giza": ["6th of October City","El Sheikh Zayed City","Al Ayyat","Al Hawamidiyya","Badrshein","El Saf","Al Omraneya","Atfih","Bahariya Oasis","Pyramids Gardens","Faisal","Haram","Abu an Numros","Al Agouzah","Al Warak","Awsīm","Boulaq Ad Dakrour","Al Doqi","Ard El Lewa","Mohandessin","Kirdasah","Imbabah","Monshaet El Kanater","Giza"],
    "Alexandria": ["Ad Dakhilah","Amreya","Borg El Arab","Moharam Bek","Bab Sharqi","Sidi Gabir","Al Attarin","Mansheya","Al Labban","El Gomrok","Port al Basal","Karmus","Montaza 2","Alexandria"],
    "Luxor": ["Luxor","Armant","Esna","Al Qarnah"],
    "Aswan": ["Aswan","Kom Ombo","Daraw","Edfu"],
    "Red Sea": ["Hurghada","Ras Ghareb","Safaga","Quseer","Marsa Alam"],
    "Assiut": ["Abnub","Abu Tig","Al Badari","El Ghanayem","El Fateh","El Quseyya","Dayrout","Sahel Selim","Sidfa","Asyut","Manfalut","Manqabad"],
    "Sohag": ["Tima","Tahta","Juhaynah","El Maragha","Sohag","Akhmim","Saqultah","Al Minshah","Girga","Al Balyana","Dar El Salam","El Usayrat"],
    "Qena": ["Qena","Qus","Madinet Qaft","Dishna","Naqadah","Nagaa Hammadi","Farshut","Abu Tesht"],
    "Minya": ["El Idwa","Maghagha","Bani Mazar","Matai","Samalut","Minya","Abu Qurqas","Mallawi","Dayr Mawas"],
    "Beni Suef": ["Al Wasta","Nasser","Beni Suef","Ihnasya Al Madinah","Biba","Sumusta El Waqf","El Fashn"],
    "Fayoum": ["Tamiya","Sinnuris","Faiyum","Atsa","Ibsheway","Youssef Al Seddik"],
    "Beheira": ["Damanhour","Kafr El Dawar","Abu Hummus","El Mahmoudeya","Natrn Valley","Al Noubareya","Abu Al Matamir","Housh Eissa","Itay Al Barud","Shubrakhit","Rosetta","Idku","Badr","Ad Dilinjat","AR Rahmaniyyah","Kom Hamada"],
    "Kafr El Sheikh": ["Kafr El Shaikh","Desouk","Fowa","Mutubas","Baltim","El Hamool","Riyadh","Biyala","Qillin","Sidi Salem"],
    "Gharbia": ["Kafr Az Zayyat","El Mahalla El Kubra","Samannoud","Basyoun","El Santa","Zefta","Kotoor","Tanta"],
    "Dakahlia": ["Mansoura","Aga","Al Manzalah","Belqas","Dikirnis","El Senbellawein","Menyet El Nasr","Mit Ghamr","Shirbin","Talkha"],
    "Sharqia": ["Bilbeis","Inshas","Awlad Saqr","El Husseiniya","Dyarb Negm","Faqus","Kafr Saqr","Mashtul as Suq","Minya Al Qamh","El Salheya El Qadima","El Salheya El Gadida","Tanis","10th of Ramadan City","Zagazig"],
    "Qalyubia": ["Banha","Al Khankah","Al Obour City","Al Qanatir Al Khayriyyah","Shibin El Qanater","Toukh","Kafr Shukr","Qalyub","Shubra Al Khaymah"],
    "Menoufia": ["Quwaysna","Birkat as SAB","El Sadat City","El Shohada","Tala","Shebeen El Kom","El Bagour","Menouf","Ashmoun"],
    "Damietta": ["Faraskur","Damietta","New Damietta","Kafr Saad","El Zarqa","Kafr Al Battikh","Ras El Bar","Izbat Al Burj"],
    "Port Said": ["Port Said"],
    "Matrouh": ["Marsa Matruh","North Coast"],
    "New Valley": ["Kharga","Dakhla","Farafra"],
    "Ismailia": ["Ismailia"],
    "Suez": ["Suez"],
    "South Sinai": ["South Sinai Governorate","Sharm El Sheikh","Dahab"],
    "North Sinai": ["North Sinai"]
}

# ----------------- helpers -----------------
def normalize_text(s: str) -> str:
    if not s: return ""
    t = s.strip()
    t = re.sub(r'[أإآ]', 'ا', t)
    t = re.sub(r'ى', 'ي', t)
    t = re.sub(r'ة', 'ه', t)  # optional normalization
    t = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', t)  # tashkeel
    # remove weird chars
    t = re.sub(r'[^\u0621-\u064Aa-zA-Z0-9\s\-\.,\/]', ' ', t)
    # arabic digits to latin
    arab_digits = '٠١٢٣٤٥٦٧٨٩'
    for i,ch in enumerate(arab_digits):
        t = t.replace(ch, str(i))
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def levenshtein(a: str, b: str) -> int:
    a,b = a.lower(), b.lower()
    if a==b: return 0
    if len(a)==0: return len(b)
    if len(b)==0: return len(a)
    v0 = list(range(len(b)+1))
    v1 = [0]*(len(b)+1)
    for i in range(len(a)):
        v1[0] = i+1
        for j in range(len(b)):
            cost = 0 if a[i]==b[j] else 1
            v1[j+1] = min(v1[j]+1, v0[j+1]+1, v0[j]+cost)
        v0, v1 = v1, v0
    return v0[len(b)]

def similarity(a: str, b: str) -> float:
    if not a and not b: return 1.0
    d = levenshtein(a,b)
    return 1.0 - (d / max(len(a), len(b), 1))

def try_match_dict(token: str, enable_fuzzy=True):
    t = token.lower().strip()
    # direct area match
    for city, areas in CITIES_DICT.items():
        for area in areas:
            if area and area.lower() == t:
                return {"city": city, "area": area, "score": 1.0, "matched_on": area}
            if area and area.lower().replace(' ', '') == t.replace(' ', ''):
                return {"city": city, "area": area, "score": 0.98, "matched_on": area}
    # fuzzy
    if enable_fuzzy:
        best = None
        for city, areas in CITIES_DICT.items():
            for area in areas:
                sc = similarity(area.lower(), t)
                if (best is None) or sc > best['score']:
                    best = {"city": city, "area": area, "score": sc, "matched_on": area}
        if best and best['score'] >= 0.65:
            return best
    # try city names
    for city in CITIES_DICT.keys():
        if city.lower() == t:
            return {"city": city, "area": city, "score": 1.0, "matched_on": city}
        if city.lower().replace(' ','') == t.replace(' ',''):
            return {"city": city, "area": city, "score": 0.98, "matched_on": city}
    if enable_fuzzy:
        bestc = None
        for city in CITIES_DICT.keys():
            sc = similarity(city.lower(), t)
            if (bestc is None) or sc > bestc['score']:
                bestc = {"city": city, "score": sc}
        if bestc and bestc['score'] >= 0.7:
            return {"city": bestc['city'], "area": bestc['city'], "score": bestc['score'], "matched_on": bestc['city']}
    return None

# attempt to prepare a transformers-based NER pipeline if model dir provided
NER_PIPELINE = None
if TRANSFORMERS_AVAILABLE:
    model_dir = os.environ.get("MODEL_DIR", "./arabert_model")  # place model in this folder if you want real token-classification
    try:
        if os.path.isdir(model_dir):
            logger.info(f"Loading token-classification model from {model_dir} ...")
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model = AutoModelForTokenClassification.from_pretrained(model_dir)
            NER_PIPELINE = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
            logger.info("Loaded local token-classification model.")
        else:
            logger.info("No local model directory found; skipping transformers NER load.")
    except Exception as e:
        logger.exception("Failed to init NER pipeline; falling back to dict matching.")

@app.post("/extract", response_model=ExtractResponse)
def extract_address(req: ExtractRequest):
    raw = req.address or ""
    if not raw.strip():
        raise HTTPException(status_code=400, detail="Empty address")
    use_fuzzy = bool(req.use_fuzzy)

    clean = normalize_text(raw)

    # If we have NER pipeline, use it first to try extract LOC/LOC tags (user can place AraBERT-trained token-classification model)
    if NER_PIPELINE:
        try:
            ner_out = NER_PIPELINE(clean)
            # collect entity text candidates labeled as LOC / B-LOC / I-LOC etc depending on model labels
            candidates = []
            for ent in ner_out:
                ent_text = ent.get('word') or ent.get('entity_group') or ent.get('entity')
                # pipeline with aggregation returns {'word','entity_group','score'}
                ent_text = ent.get('word') or ent.get('entity_group') or ent.get('entity') or ''
                # some pipelines give 'word' already chunked; use 'word' if present
                ent_text = ent.get('word', ent.get('entity', ''))
                if ent_text:
                    candidates.append(ent_text)
            # try match candidates against dict
            for cand in candidates:
                m = try_match_dict(cand, enable_fuzzy=use_fuzzy)
                if m:
                    return ExtractResponse(
                        city=m['city'],
                        area=m['area'],
                        clean_address=clean,
                        score=float(m['score']),
                        matched_on=m.get('matched_on'),
                        suggestions=[]
                    )
        except Exception:
            logger.exception("NER pipeline failed; will fallback to dict matching.")

    # If CAMEL tools available we could do normalization here (we already normalized basic)
    if CAMEL_TOOLS_AVAILABLE:
        try:
            # additional normalization (if available) - safe to call
            clean = normalize_unicode(clean)
            clean = normalize_alef_maksura_ar(clean)
        except Exception:
            pass

    # dictionary/fuzzy-based extraction: split candidate phrases and test
    # break into parts by punctuation and common Arabic prepositions
    parts = re.split(r'[,\-\/\|@]| بجوار | عند | في | داخل | بمنطقة | منطقه | منطقة ', clean, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p and p.strip()]

    best = None
    suggestions = []
    for p in parts:
        m = try_match_dict(p, enable_fuzzy=use_fuzzy)
        if m:
            suggestions.append(m)
            if (best is None) or (m['score'] > best['score']):
                best = m
        else:
            # try on n-grams within p (1..4)
            tokens = p.split()
            for n in range(min(4, len(tokens)), 0, -1):
                for i in range(0, len(tokens)-n+1):
                    gram = " ".join(tokens[i:i+n])
                    m2 = try_match_dict(gram, enable_fuzzy=use_fuzzy)
                    if m2:
                        suggestions.append(m2)
                        if (best is None) or (m2['score'] > best['score']):
                            best = m2
    # if nothing found as area, try entire cleaned string
    if not best:
        m = try_match_dict(clean, enable_fuzzy=use_fuzzy)
        if m:
            best = m
            suggestions.append(m)

    if best:
        return ExtractResponse(
            city=best['city'],
            area=best['area'],
            clean_address=clean,
            score=float(best['score']),
            matched_on=best.get('matched_on'),
            suggestions=suggestions[:5]
        )

    # nothing found
    return ExtractResponse(
        city=None,
        area=None,
        clean_address=clean,
        score=0.0,
        matched_on=None,
        suggestions=[]
    )

# health check
@app.get("/health")
def health():
    return {"ok": True, "ner_available": bool(NER_PIPELINE), "camel_tools": CAMEL_TOOLS_AVAILABLE}
