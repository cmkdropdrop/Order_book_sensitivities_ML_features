
# Order‑Book Liquidity Feature Cheat‑Sheet

*(All features are computed per raw depth snapshot and then **as‑of merged** to each 15‑minute bar, so predictors and labels refer to the same market state.)*

---

## `bid_slope`, `ask_slope`

**Definition**  
$m_{\text{bid}}$, $m_{\text{ask}}$ — slopes of the linear fits to the bid (x < 0 %) and ask (x > 0 %) wings of the snapshot depth curve.  
Units: **contracts per percent**.

**Interpretation**  
Marginal liquidity density: how much resting volume appears for each extra 1 % price concession.  
Steep = thick, flat = thin.

**Practical use**  
Quick gauge of how resilient each side of the book is to small market orders and which side currently dominates.

---

## `price_drift` ( = $x^\star$ ) & `real_liquidity` ( = $y^\star$ )

**Definition**  
Intersection of the two fitted lines  

$$
x^{\star}=\frac{b_{\text{ask}}-b_{\text{bid}}}{m_{\text{bid}}-m_{\text{ask}}}, \qquad
y^{\star}=m_{\text{bid}}\,x^{\star}+b_{\text{bid}}
$$

**Interpretation**  
$x^\star$: horizontal price shift (in %) required for bid‑side and ask‑side liquidity to balance.  
$y^\star$: cumulative inventory that must change hands to reach that balance.

**Practical use**  
Large |$x^\star$| flags directional pressure; large $y^\star$ signals a deep, healthy book that can absorb flow.

---

## `real_spread`

**Definition**  
Absolute difference between the bid and ask **x‑intercepts** (where each fitted line hits zero depth):  

$$
|x_{\text{ask}} - x_{\text{bid}}|.
$$

**Interpretation**  
A spread that includes the first chunk of hidden depth, not just the top quote.  
Wide values reveal superficially tight but shallow books.

**Practical use**  
Improves slippage models and prevents over‑optimistic cost estimates that rely on Level‑1 quotes only.

---

## `sensi_depth_vs_price_increase` & `sensi_depth_vs_price_decrease`

**Definition**  
Percentage change in depth sitting at ±1 % that would be needed to push $x^\star$ by +0.2 % (increase) or −0.2 % (decrease).  
Computed via finite‑difference bump.

**Interpretation**  
Elasticity of the book’s “centre of gravity” to cancellations or new orders near the touch.  
Low values = very twitchy; high values = stable.

**Practical use**  
Alerts to moments when tiny quote pulls could flip market sentiment or trigger stop cascades.

---

## `bid_slope_after_1pct_down`, `ask_slope_after_1pct_up`

**Definition**  
Re‑fit the impacted wing after a hypothetical 1 % market sweep removes all orders up to that level; report the new slope.

**Interpretation**  
Measures how quickly liquidity **refills** once the first layer is eaten.  
Remaining steep → robust; remaining flat → fragile.

**Practical use**  
Helps execution algos decide whether to break a large order into smaller clips or pause after an initial sweep.

---

## `sensi_price_shift_sensi_vs_AUC_buy` & `sensi_price_shift_sensi_vs_AUC_sell`

**Definition**  
Volume of market buys (or sells) needed to move $x^\star$ by +0.2 % (or −0.2 %), integrating depth in 5 bp slices outward and solving by bisection.

**Interpretation**  
Single‑number impact‑cost proxy: “how many contracts will shove the book’s balance this far?”

**Practical use**  
Direct input for order‑sizing, stress‑testing, and liquidity‑adjusted VaR calculations.

---

## At a Glance

| Family            | Features                                    | What you learn                                        |
|-------------------|---------------------------------------------|-------------------------------------------------------|
| **Immediate depth** | `bid_slope`, `ask_slope`, `real_spread`       | Thickness and true tightness right now                |
| **Balance / pressure** | `price_drift`, `real_liquidity`               | Directional bias & how much flow can rebalance it     |
| **Elasticity**    | `sensi_depth_vs_*`, `slope_after_*`         | How book shape changes when liquidity is nibbled away |
| **Impact cost**   | `sensi_price_shift_sensi_vs_AUC_*`          | Expected displacement for a given trade size          |

---
