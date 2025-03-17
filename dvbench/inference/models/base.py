from abc import ABC, abstractmethod
from typing import (
    List,
    Dict,
    Any,
    Optional,
    Type,
    Literal,
    Union,
    TypeVar,
    Generic,
    get_origin,
    get_args,
)
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    ProcessorMixin,
)
from pydantic import BaseModel, Field, ConfigDict
import importlib.util
from inspect import signature

from dvbench.inference.configs.app import TARGET_FPS


class ModelInitConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    pretrained_model_name_or_path: str = Field(
        default="please_specify_model_name", description="Name of the model"
    )
    trust_remote_code: bool = Field(
        default=True, description="Whether to trust remote code"
    )
    max_model_len: int = Field(default=4096, description="Maximum length of the model")
    max_num_seqs: int = Field(default=2, description="Maximum number of sequences")
    gpu_memory_utilization: float = Field(
        default=0.9, description="GPU memory utilization"
    )
    quantization: Optional[str] = Field(default=None, description="Quantization type")
    device: str = Field(default="cuda", description="Device")


class ModelSamplingConfig(BaseModel):
    temperature: float = Field(default=0.7, description="Temperature")
    top_p: float = Field(default=0.9, description="Top-p")
    repetition_penalty: float = Field(default=1.0, description="Repetition penalty")
    stop_token_ids: list = Field(default_factory=list)


class ModelConfig(BaseModel):
    init_config: ModelInitConfig = Field(default_factory=ModelInitConfig)
    sampling_config: ModelSamplingConfig = Field(default_factory=ModelSamplingConfig)


T = TypeVar("T", bound=ModelInitConfig)
S = TypeVar("S", bound=ModelSamplingConfig)
C = TypeVar("C", bound=ModelConfig)
P = TypeVar("P", bound=ProcessorMixin)


class ModelInfo(BaseModel):
    """Model information"""

    id: str
    modality: Literal["image", "video"]
    short_model_size: str
    full_model_size: int | None
    short_name: str
    long_name: str
    link: str
    description: str


class BaseInferenceModel(Generic[T, S, C, P]):
    """Base class for all inference models"""

    # Class-level type annotations for better readability
    init_config: T
    sampling_config: S
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
    processor: Optional[ProcessorMixin]

    def __init__(self, init_config: T, sampling_config: S, **kwargs):
        """Initialize base inference model.

        Args:
            init_config: Model initialization configuration
            sampling_config: Model sampling configuration
            **kwargs: Additional keyword arguments
        """
        # Initialize core configurations
        self._initialize_configs(init_config, sampling_config, kwargs)

        # Initialize model components
        self._initialize_components(self.init_config)

        # Set additional attributes from kwargs
        self._set_additional_attributes(kwargs)

    @staticmethod
    def _filter_params(params_dict: dict, target_class: Type) -> dict:
        """Filter parameters based on target class signature."""
        valid_params = signature(target_class).parameters.keys()
        return {k: v for k, v in params_dict.items() if k in valid_params}

    def _initialize_configs(
        self, init_config: T, sampling_config: S, kwargs: Dict[str, Any]
    ) -> None:
        """Initialize configuration objects with any overrides from kwargs."""
        # Extract config dictionaries
        init_config_dict = init_config.model_dump()
        sampling_config_dict = sampling_config.model_dump()

        # Update configs with any matching kwargs
        self._update_configs_from_kwargs(init_config_dict, sampling_config_dict, kwargs)

        # Create final config instances
        self.init_config = type(init_config)(**init_config_dict)
        self.sampling_config = type(sampling_config)(**sampling_config_dict)

    def _update_configs_from_kwargs(
        self,
        init_dict: Dict[str, Any],
        sampling_dict: Dict[str, Any],
        kwargs: Dict[str, Any],
    ) -> None:
        """Update configuration dictionaries with matching kwargs."""
        for key in list(kwargs.keys()):
            if key in init_dict:
                init_dict[key] = kwargs.pop(key)
            elif key in sampling_dict:
                sampling_dict[key] = kwargs.pop(key)

        # Log any unused parameters
        unused_keys = [key for key in kwargs.keys() if key not in ["llm", "model"]]
        if unused_keys:
            print(f"Warning: Unused parameters: {unused_keys}")

    def _initialize_components(self, init_config: T) -> None:
        """Initialize tokenizer and processor components."""
        self.tokenizer = self._load_tokenizer_static(init_config)
        self.processor = self._load_processor_static(init_config)

    def _set_additional_attributes(self, kwargs: Dict[str, Any]) -> None:
        """Set any remaining kwargs as instance attributes."""
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def get_config_class(cls, config_path, expected_config_class: C) -> C:
        """Load config class based on model name or specified path"""
        try:
            module_path = str(config_path).replace("/", ".").replace(".py", "")
            if module_path.startswith("."):
                module_path = module_path[1:]

            print("==============module_path", module_path)
            try:
                module = importlib.import_module(module_path)
            except ImportError:
                # Fallback to spec_from_file_location if direct import fails
                spec = importlib.util.spec_from_file_location(
                    config_path.stem, config_path
                )
                if spec is None or spec.loader is None:
                    raise ImportError(
                        f"Could not load module spec or loader from {config_path}"
                    )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            print("==============module", module)
            config_class = getattr(module, expected_config_class.__name__)
            if not config_class:
                raise ValueError(
                    f"No valid {expected_config_class.__name__} class found in {config_path}"
                )
        except ImportError:
            raise ImportError(f"Could not find config file at {config_path}")

        return config_class

    @staticmethod
    @abstractmethod
    def get_info() -> ModelInfo:
        """Return model information"""
        pass

    @staticmethod
    def _load_tokenizer_static(
        init_config,
    ) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        return AutoTokenizer.from_pretrained(
            init_config.pretrained_model_name_or_path,
            use_fast=init_config.use_fast,
            trust_remote_code=init_config.trust_remote_code,
        )

    @staticmethod
    def _load_processor_static(init_config) -> Optional[ProcessorMixin]:
        try:
            processor_output = AutoProcessor.from_pretrained(
                init_config.pretrained_model_name_or_path,
                trust_remote_code=init_config.trust_remote_code,
            )
            # from_pretrained might return either a processor or a tuple containing the processor and additional data
            return (
                processor_output[0]
                if isinstance(processor_output, tuple)
                else processor_output
            )
        except:
            return None

    @abstractmethod
    def prepare_inputs(
        self,
        prompt: str,
        videos: Optional[Union[List[List[Any]], List[List[str]]]] = None,
        target_fps: int = TARGET_FPS,
    ) -> Any:
        pass

    @abstractmethod
    def generate(
        self,
        prompts: Union[str, List[str]],
        videos: Optional[Union[List[List[Any]], List[List[str]]]] = None,
        **kwargs,
    ) -> List[Any]:
        pass
