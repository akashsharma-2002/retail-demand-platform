# Service Level Targets

## Targets by category
Service level is the probability of not running out of stock before the next delivery arrives.
- FOODS: 95% for FOODS_3 staple lines, 90% for FOODS_1 and FOODS_2.
- HOUSEHOLD: 90% for all departments.
- HOBBIES: 85% for all departments, because lead times are long and demand is lumpy.

## How service level maps to stock
The platform sets target stock from the upper quantile of cumulative demand over lead time plus review period.
A higher service level raises safety stock; moving from 90% to 95% typically adds 25-40% more safety stock for
slow-moving items.

## Measuring performance
Fill rate is units sold divided by units demanded. Stores report fill rate and stockout item-days weekly. A store
below 92% fill rate for two consecutive weeks is reviewed by the regional replenishment lead.
