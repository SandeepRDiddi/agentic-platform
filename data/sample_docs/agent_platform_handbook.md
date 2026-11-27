# Agent Platform Handbook

The Kill Switch can stop any agent, any tenant, or the whole platform. An administrator engages it through the AI Gateway admin API or the agentic-killswitch command. Once engaged, every agent step and every tool call is refused within one request.

Write actions such as creating a ticket never execute automatically. The Action Agent proposes the call, and a human Approver must approve it before the Tool Server will run it. The person who requested the action cannot approve it.

Model Router policy: the small model is tried first. The platform escalates to the large model when the output is invalid, truncated, or the model reports low confidence. If a provider fails, the fallback model answers.

Deployments use a canary release. Argo Rollouts sends 5 percent of traffic to the new version, runs the evaluation gate and Prometheus checks, and rolls back automatically if any check fails.

# Data Marketplace Glossary

NNDM is the Data Marketplace. A Data Product is a governed dataset published in NNDM with an owner, a contract and quality checks. The Data Steward maintains the contract. The Platform Architecture Board approves changes to the contract schema.
