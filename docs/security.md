# Security Design

## Overview

`forge-agent` is a demo-level local Agent Platform. Once an agent can call tools, it can affect the outside world through file access, knowledge base search, future shell execution, HTTP calls, database operations, and internal APIs.

Day 3 introduces the first security boundary for the platform:

* `Workspace Guard`
* `PermissionPolicy`
* structured tool errors
* permission-related trace events

The goal is not to provide a complete sandbox. The goal is to make file and knowledge-base access explicit, bounded, testable, and auditable.

## Current Security Boundary

### Workspace Root

The workspace root is the maximum file-system boundary that tools are allowed to access.

All user-provided paths must be resolved through `Workspace.resolve_user_path()` before any file-system operation is performed.

The current implementation blocks:

* path traversal such as `../secret.txt`
* nested traversal such as `docs/../../secret.txt`
* absolute paths outside the workspace
* symlink escapes from inside the workspace to files outside the workspace

The path containment check uses `Path.resolve()` and `Path.relative_to()` instead of string prefix checks.

### Why String Prefix Checks Are Not Enough

A naive implementation such as:

```python
str(path).startswith(str(root))
```

is unsafe.

For example:

```text
root = /tmp/project
path = /tmp/project-evil/secret.txt
```

The string `/tmp/project-evil/secret.txt` starts with `/tmp/project`, but it is not inside the workspace directory.

The correct approach is to resolve the path and check path hierarchy:

```python
resolved_path.relative_to(resolved_root)
```

### Permission Policy

`PermissionPolicy` is a platform-level policy layer. Tools should not each invent their own permission rules.

Current supported actions:

* `read_file`
* `write_file`
* `search_kb`
* `shell`

Default policy:

* `read_file`: allowed
* `search_kb`: allowed after workspace authorization
* `write_file`: denied unless explicitly enabled
* `shell`: denied unless explicitly enabled

Write operations are denied by default because they mutate user files and can cause irreversible project changes.

### File Tools

The current file tools are guarded:

* `list_files`
* `read_file`
* `write_file`

`list_files` and `read_file` must pass Workspace Guard before accessing the file system.

`write_file` must pass both:

1. `PermissionPolicy`
2. `Workspace Guard`

This means enabling write permission is not enough to write outside the workspace.

### RAG Tool

The RAG index path is also guarded.

`forge rag index <path>` only accepts paths inside the current workspace.

`SearchKnowledgeBaseTool` does not accept arbitrary file-system paths during execution. It searches only an already authorized `KnowledgeBase` instance.

This prevents the model from changing the knowledge base path at tool-call time.

### Structured Tool Errors

Tool failures that are expected security outcomes are represented as structured errors.

Current security error codes:

* `PERMISSION_DENIED`
* `PATH_OUTSIDE_WORKSPACE`
* `PATH_TRAVERSAL_BLOCKED`

Structured errors include:

* `error_code`
* `message`
* `reason`
* `tool_name`
* `safe_detail`

The `safe_detail` field must not contain sensitive absolute paths, secrets, tokens, or private environment details.

### Trace Events

The runtime records permission-related events for auditability.

Current permission trace events:

* `permission_check`
* `permission_denied`

A tool execution records a `permission_check` event before the tool is executed.

If a tool returns a security denial such as `PERMISSION_DENIED`, `PATH_OUTSIDE_WORKSPACE`, or `PATH_TRAVERSAL_BLOCKED`, the runtime records a `permission_denied` event.

Sensitive absolute paths in trace arguments are redacted as:

```text
<redacted-path>
```

Paths outside the workspace are represented as:

```text
<outside-workspace>
```

This keeps the trace useful for debugging without leaking sensitive file-system details.

## What This Does Not Defend Against

The current implementation is a workspace guard, not a full sandbox.

It does not defend against:

* malicious code execution inside the Python process
* unsafe shell commands if a shell tool is later enabled
* network exfiltration through future HTTP tools
* prompt injection that persuades the model to misuse allowed tools
* denial-of-service through very large files
* race conditions between path validation and file access
* operating-system-level attacks
* malicious dependencies
* full isolation between multiple users

The current implementation is appropriate for a local demo Agent Platform, but it is not a production-grade isolation boundary.

## Future Upgrade Path

### Container Sandbox

A future production-grade version should execute high-risk tools inside an isolated container.

Recommended controls:

* mount the workspace as a restricted volume
* use a non-root user
* disable privileged mode
* restrict network access by default
* apply CPU and memory limits
* apply process limits
* set read-only mounts where possible
* create a temporary working directory per run
* destroy the container after execution

### Policy Engine

The current `PermissionPolicy` is intentionally simple.

Future versions can evolve toward:

* per-tool policy
* per-path policy
* per-user policy
* per-project policy
* policy loaded from config
* policy backed by OPA or another external policy engine

### Human Approval

High-risk actions should require explicit human approval.

Tools that should require human approval:

* shell execution
* file writes outside allowlisted paths
* bulk file edits
* deleting files
* changing dependency files
* changing CI/CD files
* modifying secrets or environment files
* outbound HTTP requests to non-allowlisted domains
* database writes
* internal API calls
* deployment actions

### Better Secret Handling

Future trace and logging layers should include secret redaction for:

* API keys
* access tokens
* private keys
* passwords
* `.env` content
* authorization headers
* signed URLs
* cloud credentials

### Audit Log

Trace events are currently in-memory run artifacts.

Future versions should support durable audit logs with:

* run id
* user id
* workspace id
* tool call id
* tool name
* permission decision
* sanitized arguments
* sanitized result summary
* timestamp
* approval state
* policy version

## Current Security Guarantees

At the end of Day 3, the project can demonstrate:

* the agent can read files inside the workspace
* the agent cannot read files outside the workspace through absolute paths
* the agent cannot use `../` path traversal to escape the workspace
* the agent cannot escape through symlinks
* write operations are denied by default
* write operations require explicit permission
* RAG indexing only accepts workspace-local paths
* knowledge base search uses an authorized index
* security failures return structured tool errors
* permission checks and denials are recorded in trace events
* traces do not expose sensitive absolute paths

## Interview Summary

Agent tools are external action boundaries. Once an agent can read files, write files, call shell, or access APIs, the platform needs permission control, isolation, and auditability.

For the demo stage, `forge-agent` implements Workspace Guard and PermissionPolicy instead of a full Docker sandbox. Workspace Guard prevents path traversal, absolute path escape, and symlink escape. PermissionPolicy makes write and shell actions explicit instead of enabled by default. Security failures are returned as structured tool errors and are recorded in trace events with sanitized details.

This design is intentionally small, but it provides a clear upgrade path toward container sandboxing, human approval, durable audit logs, and policy-engine-based authorization.
