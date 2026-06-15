import os
import torch
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel

class ModelRunner:
    def __init__(self, model_dir="./math_gpt3_final"):
        """
        تهيئة مفسر الرموز والنموذج من المجلد المحفوظ.
        """
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"المجلد {model_dir} غير موجود. تأكد من مسار الحفظ الصحيح.")
            
        print(f"جاري تحميل النموذج من: {model_dir} ...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # تحميل الـ Tokenizer والنموذج
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir)
        self.model = GPT2LMHeadModel.from_pretrained(model_dir).to(self.device)
        self.model.eval() # وضع النموذج في طور التشغيل/التقييم
        
        print(f"تم تحميل النموذج بنجاح على جهاز: {self.device}\n")

    def predict(self, prompt, max_new_tokens=1024, temperature=0.6):
        """
        تأخذ السؤال وتُرجع الحل فقط بدون تكرار السؤال.
        """
        # إضافة رمز البداية إذا لم يكن موجوداً
        if "<ans>" not in prompt:
            prompt = prompt + "<ans>"
            
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # نستخدم طريقتين للتوليد: 
        # إذا كان الـ temperature منخفض جداً نستخدم Greedy Search لضمان الدقة الحسابية العالية
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True if temperature > 0 else False,
                temperature=temperature if temperature > 0 else None,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                top_p=0.9 if temperature > 0 else None
            )
            
        # فك التشفير
        full_tokens = self.tokenizer.convert_ids_to_tokens(output_ids[0])
        full_text = "".join(full_tokens)
        
        # تنظيف المخرج لعزل الإجابة فقط
        if "<ans>" in full_text:
            answer = full_text.split("<ans>")[1]
        else:
            answer = full_text
            
        # إزالة الـ <eos> والـ <pad> من النهاية لجمالية المخرج
        answer = answer.replace("<eos>", "").replace("<pad>", "").strip()
        return answer

# ==========================================
# تجربة تشغيل البرنامج كملف مستقل لتجربته فوراً
# ==========================================
if __name__ == "__main__":
    # يمكنك تغيير المسار إلى مجلد النموذج الحالي لديك لتجربته
    # مثلاً: "./math_gpt3_final"
    try:
        runner = ModelRunner(model_dir="./math_gpt3_final")
        
        print("--- برنامج حل المعادلات من الدرجة الثانية ---")
        print("اكتب 'exit' للخروج من البرنامج.\n")
        
        while True:
            user_input = input("أدخل المعادلة (مثال: 1x²-5x+6=0): ")
            if user_input.lower() == 'exit':
                break
                
            if not user_input.strip():
                continue
                
            # تشغيل النموذج
            solution = runner.predict(user_input, temperature=0.1)
            print(f"🤖 الحل المشتق من النموذج:\n{solution}")
            print("-" * 40)
            
    except Exception as e:
        print(f"حدث خطأ: {e}")