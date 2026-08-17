from typing import Dict, Any, List, Optional

class StructuredAnswerTemplates:
    """Pre-formatted deterministic template answers following AGENTS.md ideal response structure."""

    TEMPLATES = {
        "MEMBERSHIP": (
            "**للتسجيل والانتساب إلى نقابة المهندسين الأردنيين:**\n\n"
            "**الوثائق المطلوبة:**\n"
            "1. مصادقة شهادة البكالوريوس في الهندسة من وزارة التعليم العالي.\n"
            "2. صورة عن بطاقة الأحوال المدنية وجواز السفر.\n"
            "3. عدد (2) صورة شخصية.\n"
            "4. شهادة عدم محكومية.\n\n"
            "**الخطوات التالية:** قدم طلب التسجيل إلكترونياً عبر البوابة الرقمية للنقابة أو بزيارة الفرع الرئيسي."
        ),
        "FEES": (
            "**رسوم الانتساب والاشتراك السنوي:**\n\n"
            "**تفاصيل المبالغ:**\n"
            "• رسم التسجيل لأول مرة: يتم تحديده حسب الشعبة والتخصص.\n"
            "• الاشتراك السنوي: يُدفع سنوياً للحفاظ على العضوية الفعالة وصندوق التكافل.\n\n"
            "**الخطوات التالية:** يمكنك سداد الرسوم عبر تطبيق النقابة الإلكتروني أو خدمات إي فواتيركم."
        )
    }

    @classmethod
    def get_template_answer(cls, intent: str) -> Optional[str]:
        return cls.TEMPLATES.get(intent)

    @classmethod
    def format_ticket_response(cls, ticket_res: Dict[str, Any]) -> str:
        ticket_id = ticket_res.get("ticket_id", "TCK-UNKNOWN")
        category = ticket_res.get("category", "GENERAL")
        status = ticket_res.get("status", "OPEN")

        category_names = {
            "FINANCIAL": "مالية (دفع / شراء / فواتيركم)",
            "HEALTH": "تأمين صحي ومطالبات طبية",
            "TECHNICAL": "دخول البوابة وتطبيق النقابة",
            "GENERAL": "دعم واستفسارات عامة"
        }
        cat_label = category_names.get(category, category)

        return (
            f"**تم تسجيل تذكرة الدعم بنجاح!**\n\n"
            f"**تفاصيل التذكرة:**\n"
            f"• **الرقم المرجعي للتذكرة:** `{ticket_id}`\n"
            f"• **تصنيف المشكلة:** {cat_label}\n"
            f"• **حالة الطلب:** {status}\n\n"
            f"**الخطوات التالية:** تم إرسال طلبك بنجاح إلى نظام التذاكر الرئيسي، وسيقوم الفريق المعني بمتابعة طلبك."
        )
