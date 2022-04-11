import json


class TrainConfig:
    def __init__(
        self,
        model,
        dataset_name,
        teacher_model=None,
        distillation_method=None,
        output_dir="../logs",
        evaluation_strategy="steps",
        eval_steps=1000,
        save_strategy="epoch",
        save_total_limit=5,
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.01,
        optim="adamw",
        disable_tqdm=True,
        compute_metrics=False,
        metric_for_best_model="f1"   
    ):
        self.model = model
        self.dataset_name = dataset_name
        self.teacher_model = teacher_model
        self.distillation_method = distillation_method
        self.output_dir = output_dir
        self.evaluation_strategy = evaluation_strategy
        self.eval_steps = eval_steps
        self.save_strategy = save_strategy
        self.save_total_limit = save_total_limit
        self.learning_rate = learning_rate
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.num_train_epochs = num_train_epochs
        self.weight_decay = weight_decay
        self.optim = optim
        self.disable_tqdm = disable_tqdm
        self.compute_metrics = compute_metrics
        self.metric_for_best_model = metric_for_best_model
    
    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            config_dict = json.load(f)
        
        return cls(**config_dict)
