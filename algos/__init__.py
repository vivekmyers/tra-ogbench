from algos.crl import CRLAgent
from algos.gciql import GCIQLAgent
from algos.hiql import HIQLAgent
from algos.ppo import PPOAgent
from algos.sac import SACAgent

algos = dict(
    crl=CRLAgent,
    gciql=GCIQLAgent,
    hiql=HIQLAgent,
    ppo=PPOAgent,
    sac=SACAgent,
)
