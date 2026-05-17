import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
# Импортируем датасет через наш настроенный __init__.py
from src import EminemLyricsDataset


def main():

    model_id = "Qwen/Qwen2.5-1.5B-Instruct"

    print('tokenizer downloading...')

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    print(" Настройка 4-битного сжатия модели...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # Идеально для видеокарты T4 в Colab
        bnb_4bit_use_double_quant=True
    )

    # 3. Загрузка базовой модели в сжатом виде
    print("Загрузка сжатой модели...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"  # Автоматически оптимально распределит модель в памяти
    )

    # Подготавливаем веса к расчету градиентов поверх квантованных слоев
    model = prepare_model_for_kbit_training(model)

    # 4. Настройка LoRA-адаптеров
    print(" Создание структуры LoRA слоев...")
    peft_config = LoraConfig(
        r=16,                  # Ранг матриц (чем выше, тем детальнее обучение, 16 — стандарт)
        lora_alpha=32,          # Коэффициент масштабирования весов LoRA
        # Указываем, к каким именно слоям внутри Qwen мы "пришиваем" новые веса
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Оборачиваем модель: теперь обучаться будут только LoRA-слои (~1-2% от всех весов)
    model = get_peft_model(model, peft_config)
    print(" Профиль обучаемых параметров:")
    model.print_trainable_parameters()

    # 5. Инициализация нашего кастомного датасета
    print(" Подготовка данных...")
    data_path = "/home/vache/Projects/NLP/Eminem_LLM_generator/data/data.txt"
    train_dataset = EminemLyricsDataset(data_path, tokenizer, max_length=64)

    # Коллатор для автоматической сборки тензоров в батчи
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 6. Настройка гиперпараметров обучения (Production-стандарт)
    print(" Конфигурация процесса обучения...")
    training_args = TrainingArguments(
        output_dir="./eminem_qlora_result",
        per_device_train_batch_size=4,     # Маленький батч, чтобы точно не вылететь по памяти
        gradient_accumulation_steps=4,     # Шаг градиента раз в 4 батча (эффективный батч = 16)
        warmup_steps=10,                   # Плавный разогрев Learning Rate
        num_train_epochs=5,                # 5 эпох хватит для фиксации стиля куплетов
        learning_rate=2e-4,                # Оптимальный шаг для LoRA дообучения
        fp16=True,                         # Смешанная точность (ускоряет обучение в 2-3 раза)
        logging_steps=5,
        save_strategy="epoch",
        report_to="none"                   # Отключаем внешние сервисы логирования для простоты
    )

    # 7. Запуск процесса
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    print(" СТАРТ ДООБУЧЕНИЯ (Fine-tuning)...")
    trainer.train()

    # 8. Сохранение обученных адаптеров
    print(" Обучение завершено! Сохраняем только LoRA веса...")
    # Мы НЕ сохраняем гигабайты базовой Qwen, мы сохраняем только мегабайты наших адаптеров!
    trainer.model.save_pretrained("./eminem_lora_weights")
    tokenizer.save_pretrained("./eminem_lora_weights")
    print("[TRAIN] Успешно сохранено в папочку './eminem_lora_weights'")

if __name__ == "__main__":
    main()