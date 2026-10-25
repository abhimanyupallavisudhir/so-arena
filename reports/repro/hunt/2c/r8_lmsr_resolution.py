"""MarketScoringReward looks the resolution up as a label; ResolutionScore (same module family, same
outcome.data['resolution'] key) documents 1/0 or True/False. With a 1/0 or 'YES' resolution every
trade scores log(eps)-log(eps)=0: all traders silently get 0.0 (status final), not None or an error."""
from so_arena.core.mechanism import Episode, Outcome
from so_arena.core.rewards import ResolutionScore
from so_arena.mechanisms.markets import MarketScoringReward

trades = [{"role": "trader_1", "before": {"yes": .5, "no": .5}, "after": {"yes": .9, "no": .1}},
          {"role": "trader_2", "before": {"yes": .9, "no": .1}, "after": {"yes": .2, "no": .8}}]
for y in ("yes", 1, True, "YES"):
    ep = Episode(id="e", item_id="i", mechanism="prediction_market",
                 outcome=Outcome(data={"trades": trades, "trader_roles": ["trader_1", "trader_2"], "resolution": y,
                                       "forecasts": {"trader_1": .9, "trader_2": .2}}),
                 trainable_roles=["trader_1", "trader_2"])
    print(f"resolution={y!r:6}: LMSR {MarketScoringReward().compute(ep)}  | ResolutionScore {ResolutionScore().compute(ep)}")
