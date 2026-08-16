from pydantic import BaseModel, ConfigDict, Field

from research_gap_agent.tools import ToolRegistry


class EchoInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)


def test_registry_exposes_function_schema_and_dispatches_validated_input() -> None:
    registry = ToolRegistry()
    registry.register(
        name="echo_text",
        description="Return validated text.",
        input_model=EchoInput,
        handler=lambda value: {"text": value.text},
    )

    schema = registry.function_schemas()[0]
    result = registry.invoke("echo_text", {"text": "hello"})

    assert schema["type"] == "function"
    assert schema["function"]["name"] == "echo_text"
    assert schema["function"]["parameters"]["additionalProperties"] is False
    assert result.ok is True
    assert result.output == {"text": "hello"}


def test_registry_returns_structured_validation_and_unknown_tool_errors() -> None:
    registry = ToolRegistry()
    registry.register(
        name="echo_text",
        description="Return validated text.",
        input_model=EchoInput,
        handler=lambda value: value,
    )

    invalid = registry.invoke("echo_text", {"text": "", "extra": True})
    missing = registry.invoke("not_registered", {})

    assert invalid.ok is False
    assert invalid.error_type == "validation_error"
    assert missing.ok is False
    assert missing.error_type == "tool_not_found"


def test_registry_rejects_duplicate_tool_names() -> None:
    registry = ToolRegistry()
    registry.register(
        name="echo_text",
        description="Return validated text.",
        input_model=EchoInput,
        handler=lambda value: value,
    )

    try:
        registry.register(
            name="echo_text",
            description="Duplicate.",
            input_model=EchoInput,
            handler=lambda value: value,
        )
    except ValueError as exc:
        assert "already registered" in str(exc)
    else:
        raise AssertionError("expected duplicate registration to fail")
