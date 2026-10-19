"""v1.2.5 changes only fixed TCP local Y to an absolute 0.5 metres."""
from .nandu1_v1_2_3_mdp import Nandu1FixedTCPHeightCommand
from .nandu1_v1_2_tcp_contract import LONG_TCP_LOCAL


class Nandu1LongTCPHeightCommand(Nandu1FixedTCPHeightCommand):
    TCP_LOCAL = LONG_TCP_LOCAL
