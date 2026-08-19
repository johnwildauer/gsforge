# Software Forge

`.forge/` is the portable, Markdown-native workflow for agent-assisted software development. It is the source of truth for the workflow itself, the durable understanding of a host repository, and the record of bounded implementation initiatives.

## Start here

1. Read [`process/00-overview.md`](process/00-overview.md).
2. Determine whether this is a new repository or an existing repository.
3. For a new repository, follow [`process/01-init-new.md`](process/01-init-new.md).
4. For an existing repository, follow [`process/01-init-existing.md`](process/01-init-existing.md).
5. If the repository has already been initialized, use [`process/02-initiative.md`](process/02-initiative.md) to begin a new initiative.

Do not edit workflow artifacts until you have read the process document governing the current step. Do not rely on temporary files outside `.forge/` when operating in an initialized host repository.

## Directory map

| Directory | Purpose |
|---|---|
| [`process/`](process/) | Agent instructions for initialization, planning, design review, implementation, implementation review, and closeout. |
| [`requirements/`](requirements/) | Durable product intent, user-facing capabilities, and testable acceptance criteria. |
| [`blueprints/`](blueprints/) | Durable technical context, architecture, component behavior, contracts, and feature composition. |
| [`initiatives/`](initiatives/) | Bounded change sets containing plans, work orders, reviews, and execution history. |

## Core boundaries

- **Requirements** describe what users and stakeholders need and why. They should not prescribe internal implementation.
- **Blueprints** describe how the system is structured and how it satisfies requirements.
- **Initiatives** coordinate a specific change. They are not a replacement for the durable requirements or blueprints.

## Process order

Initialization establishes the repository baseline. Each later initiative follows this order:

1. Plan and iterate on intent in [`process/02-initiative.md`](process/02-initiative.md).
2. Review the proposed design with [`process/03-design-review.md`](process/03-design-review.md).
3. Execute work orders using [`process/04-work-order.md`](process/04-work-order.md).
4. Validate implementation using [`process/05-implementation-review.md`](process/05-implementation-review.md).
5. Reconcile the branch and close the initiative using [`process/06-closeout-review.md`](process/06-closeout-review.md).

The project-specific policy is maintained in the policy section of [`process/00-overview.md`](process/00-overview.md). It may add stricter gates or commands, but later process documents must not silently weaken it.
