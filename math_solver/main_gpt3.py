import random
import math
import torch
from torch.utils.data import Dataset
from sympy import symbols, expand
from tokenizers import Tokenizer, models, pre_tokenizers, decoders
from transformers import (
    GPT2Config, GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)

# =========================
# 1. DATA GENERATION
# =========================

x = symbols("x")

def generate_data_with_steps(n=25000):
    data = []
    
    while len(data) < n:
        # 1. اختيار المعاملات عشوائياً (لتجنب التعقيد سنركز على المعاملات التي تعطيك دلتا موجب)
        a = random.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
        b = random.randint(-15, 15)
        c = random.randint(-15, 15)
        
        # حساب الدلتا حاسوبياً أولاً للتحقق
        delta = b**2 - 4*a*c
        
        # سنكتفي حالياً بالمعادلات التي تملك حلولاً حقيقية (delta >= 0) لتسهيل التعلم على النموذج
        if delta < 0:
            continue
            
        sqrt_d = math.sqrt(delta)
        
        # حساب الجذور وتقريبها لمرتبة عشرية واحدة أو اثنتين لتسهيل النص
        x1 = round((-b + sqrt_d) / (2 * a), 2)
        x2 = round((-b - sqrt_d) / (2 * a), 2)
        
        # 2. صياغة السؤال (المعالمة) بشكل نظيف مع الإشارات
        # استخدام b:+ و c:+ يضمن ظهور إشارة الزائد تلقائياً للأعداد الموجبة
        question = f"{a}x²{b:+}x{c:+}=0"
        
        # 3. صياغة "خطوات الحل التفصيلية" (Chain of Thought)
        # سنقسمها إلى خطوات واضحة يفهم النموذج تتابعها المنطقي
        step_a_b_c = f"a={a};b={b};c={c}"
        
        # خطوة حساب الدلتا: القانون -> التعويض -> الناتج النهائي
        step_delta = f"d=({b})²-4*({a})*({c})={b**2}-({4*a*c})={delta}"
        
        # خطوة حساب الجذور بالتفصيل
        if delta > 0:
            step_roots = f"x1=(-({b})+{round(sqrt_d, 2)})/(2*{a})={x1};x2=(-({b})-{round(sqrt_d, 2)})/(2*{a})={x2}"
        else: # delta == 0
            step_roots = f"x1=x2=-({b})/(2*{a})={x1}"
            
        # دمج كل الخطوات بفواصل واضحة (مثلاً استخدام كلمة step أو رموز ثابتة)
        answer = f"{step_a_b_c}|step1:{step_delta}|step2:{step_roots}<eos>"
        
        # إضافة السلسلة الكاملة للنص
        data.append(question + "<ans>" + answer)
        
    return data

# معاينة سريعة لشكل البيانات الجديدة
texts = generate_data_with_steps(n=3)
for text in texts:
    print(text)
    print("-" * 50)

texts = generate_data_with_steps(n=25000) # <--- هذا السطر الذي كان ينقصك!
# =========================
# 2. BUILD TOKENIZER
# =========================

special_tokens = ["<pad>", "<eos>", "<ans>"]

# Collect all characters and special tokens
chars = set()
for t in texts:
    chars.update(list(t))
chars.update(special_tokens)
chars = sorted(chars)

# Create vocabulary mapping
vocab = {c: i for i, c in enumerate(chars)}

# Build a WordLevel tokenizer with this vocab
base_tokenizer = Tokenizer(models.WordLevel(vocab, unk_token=None))
base_tokenizer.pre_tokenizer = pre_tokenizers.Split("", behavior="removed")  # character-level


# Wrap into PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=base_tokenizer,
    pad_token="<pad>",
    eos_token="<eos>",
    unk_token=None,
    additional_special_tokens=["<ans>"]
)

# Add special tokens (they already exist, but ensures proper config)
tokenizer.add_special_tokens({"pad_token": "<pad>", "eos_token": "<eos>", "additional_special_tokens": ["<ans>"]})

PAD_IDX = tokenizer.pad_token_id
EOS_IDX = tokenizer.eos_token_id
VOCAB_SIZE = tokenizer.vocab_size

print(f"Vocabulary size: {VOCAB_SIZE}")
print(f"PAD ID: {PAD_IDX}, EOS ID: {EOS_IDX}")

# Quick test
test_enc = tokenizer.encode("1x²-5x+6=0<ans>")
print("Encoded tokens:", tokenizer.convert_ids_to_tokens(test_enc))
print("Decoded:", tokenizer.decode(test_enc))

# =========================
# 3. DATASET
# =========================

class MathDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)
        labels = input_ids.clone()
        
        # استبدال معرف الـ pad بـ -100 لحجبه عن الـ Loss
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {"input_ids": input_ids, "labels": labels}

dataset = MathDataset(texts, tokenizer, max_length=128)

# =========================
# 4. MODEL
# =========================

config = GPT2Config(
    vocab_size=VOCAB_SIZE,
    n_positions=128,
    n_embd=512,
    n_layer=6,
    n_head=8,
    pad_token_id=PAD_IDX,
    eos_token_id=EOS_IDX,
    bos_token_id=None,
    n_inner=1024
)

model = GPT2LMHeadModel(config)
print(f"Model parameters: {model.num_parameters()}")

# =========================
# 5. TRAINING (بدون overwrite_output_dir)
# =========================

training_args = TrainingArguments(
    output_dir="./math_gpt3",
    num_train_epochs=8,
    per_device_train_batch_size=32,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_steps=500,
    logging_steps=100,
    save_steps=500,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    report_to="none",
    remove_unused_columns=False,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

print("Starting training...")
trainer.train()

# Save model
trainer.save_model("./math_gpt3_final")
tokenizer.save_pretrained("./math_gpt3_final")

# =========================
# 6. GENERATION
# =========================

def generate(prompt, max_new_tokens=60, temperature=0.7):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        eos_token_id=EOS_IDX,
        pad_token_id=PAD_IDX,
        top_k=0,
        top_p=0.9
    )
    # فك التشفير يدوياً دون مسافات
    tokens = tokenizer.convert_ids_to_tokens(output_ids[0])
    text = "".join(tokens)           # يدمج الحروف كما هي
    return text

# =========================
# 7. TEST
# =========================

print("\nGenerated:")
print(generate("2x²+3x-2=0<ans>"))

# =========================
# اختبارات جاهزة بعد التدريب
# =========================

test_prompts = [
    # معادلات بسيطة (a=1)
    "1x²-5x+6=0<ans>",      # x1=2, x2=3
    "1x²+0x-4=0<ans>",      # x1=2, x2=-2
    "1x²-7x+12=0<ans>",     # x1=3, x2=4
    "1x²+3x+2=0<ans>",      # x1=-1, x2=-2

    # معادلات بمعامل a ≠ 1
    "2x²-5x+2=0<ans>",      # x1=2, x2=0.5 (أو كسور)
    "3x²-7x+2=0<ans>",      # x1=2, x2=1/3
    "4x²-4x+1=0<ans>",      # x1=0.5, x2=0.5 (جذر مزدوج)

    # معادلات بمعامل سالب
    "-1x²+5x-6=0<ans>",     # نفس جذور x²-5x+6 لكن بإشارة معكوسة
    "-2x²+8x-6=0<ans>",     # x1=1, x2=3

    # حالة المميز صفر
    "1x²-4x+4=0<ans>",      # x1=2, x2=2

    # معادلات عشوائية إضافية
    "1x²-3x+2=0<ans>",      # x1=1, x2=2
    "2x²+3x-2=0<ans>",      # x1=0.5, x2=-2 (اختبارك السابق)
]

print("=" * 60)
print("اختبار النموذج على معادلات متنوعة:")
print("=" * 60)

for prompt in test_prompts:
    generated = generate(prompt, max_new_tokens=120, temperature=0.7)
    print(f"\nالمدخل: {prompt}")
    print(f"المخرج: {generated}")
    print("-" * 60)