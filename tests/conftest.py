import sys
from pathlib import Path

import pytest

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

@pytest.fixture
def sample_queries():
    return {
        "registration_msa": "ما هي شروط تسجيل المهندسين الأردنيين خريجي الجامعات الأردنية؟",
        "registration_jordanian": "شو الأوراق المطلوبة عشان أسجل بالنقابة أول مرة؟",
        "salary_msa": "ما هو سلم رواتب المهندسين والحد الأدنى للأجور؟",
        "salary_jordanian": "قديش الراتب الأساسي لمهندس حديث التخرج؟",
        "accreditation_msa": "ما هي شروط الحصول على مرتبة مهندس مستشار؟",
        "english_query": "What are the requirements for joining the Jordan Engineers Association?"
    }

@pytest.fixture
def sample_documents():
    return [
        {
            "id": "doc1",
            "title": "شروط تسجيل المهندسين الأردنيين",
            "text": "الوثائق المطلوبة لتسجيل المهندسين الأردنيين: هوية الأحوال المدنية المصدقة، كشف علامات الثانوية العامة، الشهادة الجامعية الأصلية، ورسم التسجيل 136 دينار أردني.",
            "source": "شروط-تسجيل-الاردنيين.md",
            "similarity_score": 92.5,
            "rank": 1
        },
        {
            "id": "doc2",
            "title": "سلم رواتب المهندسين",
            "text": "الحد الأدنى لرواتب المهندسين حديثي التخرج من سنة 0 إلى سنة 1 هو 400 دينار أردني، ويصل إلى 1350 دينار للمزيد من 12 سنة خبرة.",
            "source": "سلم_الرواتب.md",
            "similarity_score": 88.0,
            "rank": 2
        },
        {
            "id": "doc3",
            "title": "نظام التأهيل والاعتماد المهني",
            "text": "مراتب الاعتماد المهني هي: مهندس، مهندس مشارك، مهندس محترف، ومهندس مستشار. يتطلب المهندس المستشار خبرة 10 سنوات بعد المهندس المحترف.",
            "source": "نظام_التأهيل.md",
            "similarity_score": 81.2,
            "rank": 3
        }
    ]
