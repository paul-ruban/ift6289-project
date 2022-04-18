import json


class TrainConfig:
    def __init__(
        self,
        do_train,
        model,
        dataset_name,
        teacher_model=None,
        distillation_method=None,
        output_dir="../logs",
        evaluation_strategy="steps",
        eval_steps=1000,
        save_strategy="epoch",
        save_steps=1000,
        save_total_limit=5,
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.01,
        optim="adamw",
        disable_tqdm=True,
        compute_metrics=False,
        metric_for_best_model="loss",
        pruning_config=None,
        quantize = None
    ):
        """ TrainConfig
        
        Args:
            do_train (bool): Whether to train the model.
            model (str): model name
            dataset_name (str): dataset name
            teacher_model (str): teacher model name
            distillation_method (str): distillation method name
            output_dir (str): output directory
            evaluation_strategy (str): evaluation strategy
            eval_steps (int): evaluation steps
            save_strategy (str): save strategy
            save_steps (int): save steps
            save_total_limit (int): save total limit
            learning_rate (float): learning rate
            per_device_train_batch_size (int): per device train batch size
            per_device_eval_batch_size (int): per device eval batch size
            num_train_epochs (int): num train epochs
            weight_decay (float): weight decay
            optim (str): optimizer
            disable_tqdm (bool): disable progress bar
            compute_metrics (bool): compute metrics
            metric_for_best_model (str): metric for best model
            pruning_config (dict): pruning config
            quantize (None, str): quantize
        """
        self.do_train = do_train
        self.model = model
        self.dataset_name = dataset_name
        self.teacher_model = teacher_model
        self.distillation_method = distillation_method
        self.output_dir = output_dir
        self.evaluation_strategy = evaluation_strategy
        self.eval_steps = eval_steps
        self.save_strategy = save_strategy
        self.save_steps = save_steps
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
        self.pruning_config = pruning_config
        self.quantize = quantize

    @classmethod
    def from_json(cls, path):
        """ Load from json file

        Args:
            path (str): path to json file
        
        Returns:
            TrainConfig object
        """
        with open(path) as f:
            config_dict = json.load(f)
        
        return cls(**config_dict)
