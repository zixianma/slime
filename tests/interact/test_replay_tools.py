from examples.interact.tools.visualization.replay_common import (
    cooking_response,
    cooking_task_instruction,
    cooking_user_text,
)


def test_cooking_replay_text_parsing():
    text, flag, progress = cooking_response(
        'prefix {"text":"Use the cheese next","flag":{"kind":"repeat"},"cook_progress":3}<|im_end|>'
    )
    assert text == "Use the cheese next"
    assert flag == {"kind": "repeat"}
    assert progress == 3
    assert cooking_user_text(
        'state\n== THE COOK JUST SAID ==\n"I found it"\n\nBelow are the views'
    ) == "I found it"
    assert cooking_task_instruction(
        "The cook was asked to make: a cheeseburger\nAdditional rules"
    ) == "a cheeseburger"
