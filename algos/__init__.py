from algos.crl import CRLAgent
from algos.gcbc import GCBCAgent
from algos.gciql import GCIQLAgent
from algos.hiql import HIQLAgent
from algos.ppo import PPOAgent
from algos.qrl import QRLAgent
from algos.sac import SACAgent
from algos.tra import TRAAgent

algos = dict(
    crl=CRLAgent,
    gcbc=GCBCAgent,
    gciql=GCIQLAgent,
    hiql=HIQLAgent,
    ppo=PPOAgent,
    qrl=QRLAgent,
    sac=SACAgent,
    tra=TRAAgent,
)
