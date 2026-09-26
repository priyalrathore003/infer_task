"""Offline tests: real tau2 retail env + real LiveKit AgentSession, scripted LLM."""

from tau2.data_model.message import AssistantMessage, UserMessage
from tau2.runner import build_environment

from voice_tau.bridge import LiveKitTauAgent
from voice_tau.config import RunCfg

from .fake_llm import ScriptedLLM


def _agent(script):
    env = build_environment("retail")
    a = LiveKitTauAgent(env.get_tools(), env.get_policy(), RunCfg())
    a._llm_override = ScriptedLLM(script)
    return env, a


def test_tool_roundtrip_through_tau2_env():
    env, agent = _agent(
        [
            {"tool": "find_user_id_by_name_zip", "args": {"first_name": "Yusuf", "last_name": "Rossi", "zip": "19122"}},
            {"tool": "get_user_details", "args": {"user_id": "yusuf_rossi_9620"}},
            {"text": "Thanks Yusuf, I found your account. How can I help?"},
        ]
    )
    greeting = AssistantMessage.text(content="Hi! How can I help you today?")
    state = agent.get_init_state([greeting])
    try:
        # user turn -> first tool call surfaces to tau2
        m, state = agent.generate_next_message(
            UserMessage.text(content="I'm Yusuf Rossi, zip 19122"), state
        )
        assert m.is_tool_call() and m.tool_calls[0].name == "find_user_id_by_name_zip"

        # tau2 executes against ITS env, feeds result back -> second tool call
        tool_msg = env.get_response(m.tool_calls[0])
        assert not tool_msg.error and "yusuf_rossi" in tool_msg.content
        m, state = agent.generate_next_message(tool_msg, state)
        assert m.tool_calls[0].name == "get_user_details"

        m, state = agent.generate_next_message(env.get_response(m.tool_calls[0]), state)
        assert not m.is_tool_call() and "Yusuf" in m.content

        # LiveKit saw the greeting + tool outputs in its chat context
        ctx = agent._llm_override.seen_ctx[-1]
        roles = [getattr(i, "role", i.type) for i in ctx.items]
        assert "function_call_output" in roles
        assert any(getattr(i, "text_content", None) == greeting.content for i in ctx.items)
    finally:
        agent.stop()


def test_tool_error_is_reported_to_llm():
    env, agent = _agent(
        [
            {"tool": "get_order_details", "args": {"order_id": "#W0000000"}},
            {"text": "I couldn't find that order."},
        ]
    )
    state = agent.get_init_state()
    try:
        m, state = agent.generate_next_message(UserMessage.text(content="order #W0000000?"), state)
        res = env.get_response(m.tool_calls[0])
        assert res.error
        m, state = agent.generate_next_message(res, state)
        assert "couldn't" in m.content
        out = [i for i in agent._llm_override.seen_ctx[-1].items if i.type == "function_call_output"]
        assert out and out[-1].is_error
    finally:
        agent.stop()
