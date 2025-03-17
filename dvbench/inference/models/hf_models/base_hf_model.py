from typing import cast, Union, get_origin, get_args
from abc import abstractmethod
from dvbench.inference.models.base import BaseInferenceModel, T, S, C, P
from transformers import PreTrainedModel


class BaseHFModel(BaseInferenceModel[T, S, C, P]):
    """Base class for Hugging Face models"""

    model: PreTrainedModel

    def __init__(self, config_class_or_path: Union[C, str], **kwargs):
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

        init_config: T = cast(T, config_class.init_config)
        sampling_config: S = cast(S, config_class.sampling_config)

        # Initialize HF-specific model
        self.model = self._initialize_model_static(init_config)
        if self.model:
            self.model: PreTrainedModel = cast(PreTrainedModel, self.model)
            self.model = self.model.to(init_config.device)  # type: ignore

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
            if get_origin(base) is BaseHFModel:
                type_args = get_args(base)
                if len(type_args) >= 3:  # We need at least T, S, C
                    return type_args[2]

        raise TypeError(
            "Could not determine expected config class type. "
            "Make sure the class properly inherits from BaseHFModel with generic parameters"
        )

    @staticmethod
    @abstractmethod
    def _initialize_model_static(init_config) -> PreTrainedModel:
        pass
