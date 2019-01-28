from entries.template import create_build_f, create_env_f, create_action_space_f
from entries.q_template import create_q_entry
from models import dom_qnet


main = create_q_entry(create_build_f, create_env_f, create_action_space_f)

"""
main = create_miniwob_dqn_main(
        dom_qnet.Qnet,
        is_graph=True
        )
"""

