from pydantic import ValidationError
import pytest

from llm_graph.messages import BaseMessage, AIMessage, HumanMessage

def test_valid_base_message():
    msg = BaseMessage(content="Hi",role="User")
    assert msg.content == "Hi"
    assert msg.role == "User"

def test_missing_content():
    with pytest.raises(ValidationError) as exc_info:
        msg = BaseMessage(role="User")

    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]['loc'] == ('content',)
    assert errors[0]['type'] == 'missing'
    assert 'Field required' in errors[0]['msg']

def test_missing_role():
    with pytest.raises(ValidationError) as exc_info:
        msg = BaseMessage(content="Hello there")

    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]['loc'] == ('role',)
    assert errors[0]['type'] == 'missing'
    assert 'Field required' in errors[0]['msg']

def test_none_content():
    with pytest.raises(ValidationError) as exc_info:
        msg = BaseMessage(content=None,role="User")
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]['loc'] == ('content',)
    assert errors[0]['type'] == 'string_type'
    assert 'Input should be a valid string' in errors[0]['msg']

def test_none_role():
    with pytest.raises(ValidationError) as exc_info:
        msg = BaseMessage(content="hi there",role=None)
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]['loc'] == ('role',)
    assert errors[0]['type'] == 'string_type'
    assert 'Input should be a valid string' in errors[0]['msg']

def test_invalid_role_type():
    with pytest.raises(ValidationError) as exc_info:
        msg = BaseMessage(content="hi there",role=123)
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]['loc'] == ('role',)
    assert errors[0]['type'] == 'string_type'
    assert 'Input should be a valid string' in errors[0]['msg']

def test_invalid_content_type():
    with pytest.raises(ValidationError) as exc_info:
        msg = BaseMessage(content=123,role="User")
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]['loc'] == ('content',)
    assert errors[0]['type'] == 'string_type'
    assert 'Input should be a valid string' in errors[0]['msg']

def test_dict_function():
    msg = BaseMessage(content="Hi",role="User")
    msg_dict = msg.to_dict()

    test_msg_result = {
            "content" : "Hi",
            "role" : "User"
            }
    assert test_msg_result == msg_dict
