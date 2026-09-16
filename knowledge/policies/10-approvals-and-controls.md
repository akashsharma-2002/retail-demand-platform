# Approvals and Controls

## Roles
- Viewer: can see forecasts, stock and plans.
- Planner: can create order plans and what-if scenarios.
- Approver: can approve or reject plans.
- Admin: manages users and configuration.

## Separation of duties
The person who creates a plan cannot approve it. The system enforces this for every plan.

## Approval limits
Store approvers can approve plans up to $50,000 per week per store. Plans above that need the regional finance partner.

## Audit
Every plan creation, approval and rejection is written to an append-only audit log with user, time, before and after
status and request id. Audit records are kept for 7 years.

## Assistant
The planning assistant can explain forecasts and plans and prepare orders, but it cannot submit an order. Only an
approver can release an order.
