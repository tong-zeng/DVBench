from dvbench.inference.models.base import (
    BaseInferenceModel,
    ModelConfig,
    Generic,
    T,
    S,
    C,
    P,
)
from vllm import LLM, SamplingParams
from typing import (
    List,
    Optional,
    Any,
    Union,
    cast,
    get_args,
    get_origin,
)
from abc import abstractmethod
from dvbench.inference.configs.app import TARGET_FPS
from typing import Type
from typing import get_origin, get_args, Type, TypeVar, Union
from vllm.engine.arg_utils import EngineArgs


class BaseVLLMModel(BaseInferenceModel[T, S, C, P], Generic[T, S, C, P]):
    """Base class for VLLM models"""

    # Class-level type annotations
    llm: LLM

    def __init__(self, config_class_or_path: Union[C, str], **kwargs):
        """Initialize VLLM model.

        Args:
            config_class_or_path: Either a config class instance or path to config
            **kwargs: Additional keyword arguments

        Raises:
            TypeError: If the provided config class doesn't match the expected type C
        """
        # Get the expected config class type from generic parameter C
        expected_config_class = cast(C, self._get_expected_config_class())

        # Load config class if path provided
        config_class = (
            self.get_config_class(config_class_or_path, expected_config_class)
            if isinstance(config_class_or_path, str)
            else config_class_or_path
        )
        config_class = cast(C, config_class)
        config_model = (
            config_class() if isinstance(config_class, type) else config_class
        )

        # Convert schemas to dictionaries and compare their keys
        if not (
            hasattr(config_model, "model_dump")
            and hasattr(expected_config_class, "model_dump")
            and set(config_model.model_dump().keys())
            == set(expected_config_class().model_dump().keys())  # type: ignore
        ):
            raise TypeError(
                f"Config class must have the same fields as {expected_config_class.__name__}. "
                f"Expected fields: {set(expected_config_class().model_dump().keys())}, "  # type: ignore
                f"got: {set(config_model.model_dump().keys())}"
            )

        # Rest of the initialization code...
        init_config: T = cast(T, config_model.init_config)
        sampling_config: S = cast(S, config_model.sampling_config)

        self.llm = self._initialize_llm_static(init_config)
        self.expected_config_class = expected_config_class

        super().__init__(
            init_config=init_config, sampling_config=sampling_config, **kwargs
        )

    @classmethod
    def _get_expected_config_class(cls) -> C:
        """Get the expected config class type from generic parameter C.

        Returns:
            The expected config class type

        Raises:
            TypeError: If the generic type C cannot be determined
        """
        # Get the generic parameters of the current class
        for base in cls.__orig_bases__:  # type: ignore
            if get_origin(base) is BaseVLLMModel:
                type_args = get_args(base)
                if len(type_args) >= 3:  # We need at least T, S, C
                    return type_args[2]

        raise TypeError(
            "Could not determine expected config class type. "
            "Make sure the class properly inherits from BaseVLLMModel with generic parameters"
        )

    @staticmethod
    def _initialize_llm_static(init_config) -> LLM:
        config_dict = init_config.model_dump()
        llm_class_params = BaseVLLMModel._filter_params(config_dict, LLM)
        llm_engine_params = BaseVLLMModel._filter_params(config_dict, EngineArgs)
        kwargs = {**llm_class_params, **llm_engine_params}
        return LLM(
            model=init_config.pretrained_model_name_or_path,
            max_model_len=kwargs.pop("max_model_len"),
            trust_remote_code=kwargs.pop("trust_remote_code"),
            gpu_memory_utilization=kwargs.pop("gpu_memory_utilization"),
            quantization=kwargs.pop("quantization"),
            device=kwargs.pop("device"),
            **kwargs,
        )

    def get_sampling_params(self) -> SamplingParams:
        sampling_config_dict = self.sampling_config.model_dump()
        sampling_params = BaseVLLMModel._filter_params(
            sampling_config_dict, SamplingParams
        )
        return SamplingParams(**sampling_params)

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
