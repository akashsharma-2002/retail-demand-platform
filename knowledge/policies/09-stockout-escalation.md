# Stockout Escalation

## Detection
A stockout is recorded when an item has demand and zero stock on hand. The platform counts stockout item-days.

## Escalation levels
- Level 1: one store, one item, under 3 days. Handled by the planner at the next review.
- Level 2: an item out in 2 or more stores, or any item out for 3+ days. Planner raises an emergency order.
- Level 3: a top-100 item out in all stores. Regional replenishment lead contacts the supplier the same day.

## Substitution
When a FOODS item is out, stores may raise the order for its listed substitute by up to 30% for one review.
