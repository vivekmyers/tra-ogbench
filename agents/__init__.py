from agents.crl import CRLAgent
from agents.gcbc import GCBCAgent
from agents.gciql import GCIQLAgent
from agents.hiql import HIQLAgent
from agents.ppo import PPOAgent
from agents.qrl import QRLAgent
from agents.sac import SACAgent
from agents.cmd import CMDAgent
from agents.tra import TRAAgent

agents = dict(
    crl=CRLAgent,
    gcbc=GCBCAgent,
    gciql=GCIQLAgent,
    hiql=HIQLAgent,
    ppo=PPOAgent,
    qrl=QRLAgent,
    sac=SACAgent,
    cmd=CMDAgent,
    tra=TRAAgent,
)
