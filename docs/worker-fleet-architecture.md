# Worker fleet architecture

The worker fleet is a pool of detached host processes that compete to consume
classification jobs from a durable RabbitMQ queue. The `screencrop-supervisorctl`
CLI spawns and monitors these workers, assigns each a unique metrics port and log
path, and coordinates warm shutdown by draining in-flight work before exit. Jobs
move through Postgres from `pending` (API-created) → `processing` (worker-claimed)
→ `done` or `failed`, while metrics flow to Prometheus and Grafana.

## Fleet topology

The CLI posts compressed screenshots to a stateless FastAPI service, which writes
jobs to Postgres and publishes them to the shared queue. Workers operate as
competing consumers — RabbitMQ round-robins deliveries to whichever is free. All
components emit metrics that Prometheus scrapes and Grafana visualizes.

```mermaid
graph LR
    cli["screencrop-cli"]
    api["FastAPI<br/>:8000"]
    pg[("Postgres<br/>:5432")]
    mq{{"RabbitMQ Queue<br/>screennet_inference_queue<br/>:5672"}}
    prom["Prometheus<br/>:9091"]
    graf["Grafana<br/>:3001"]

    subgraph Fleet["Fleet (host)"]
        sup["supervisorctl<br/>screencrop-supervisorctl"]
        w1["worker@1<br/>:8001"]
        w2["worker@2<br/>:8002"]
        w3["worker@3<br/>:8003"]
        sup -->|spawn/monitor| w1
        sup -->|spawn/monitor| w2
        sup -->|spawn/monitor| w3
    end

    subgraph Observability["Observability"]
        prom
        graf
    end

    cli -->|POST WebP| api
    api -->|write pending| pg
    api -->|publish| mq
    mq -->|competing consumers| w1
    mq -->|competing consumers| w2
    mq -->|competing consumers| w3
    w1 -->|write result| pg
    w2 -->|write result| pg
    w3 -->|write result| pg
    api -->|scrape| prom
    w1 -->|scrape| prom
    w2 -->|scrape| prom
    w3 -->|scrape| prom
    prom -->|query| graf
```

## Warm-shutdown handshake

When `stop` or `restart` is invoked, the supervisor sends SIGTERM to each worker.
The worker immediately cancels its RabbitMQ consumer to block new deliveries, then
drains in-flight handlers up to `shutdown_timeout` (default 30s). Messages that
finish are acknowledged; overruns stay unacked and are requeued by RabbitMQ to
another free worker (at-least-once delivery). Once drained, the worker closes the
connection. The supervisor polls PID liveness and SIGKILLs any straggler, then
clears the state file.

```mermaid
sequenceDiagram
    participant Sup as supervisorctl
    participant W as worker@i
    participant MQ as RabbitMQ

    Sup ->> W: SIGTERM
    W ->> MQ: cancel consumer (no new deliveries)
    Note over W: drain in-flight up to shutdown_timeout
    alt in-flight finishes
        W ->> MQ: ack
    else overruns
        Note over W,MQ: message stays unacked
        Note over MQ: requeues to another worker
    end
    W ->> W: close connection
    Sup ->> Sup: poll PID liveness up to --timeout
    opt still alive
        Sup ->> W: SIGKILL
    end
    Sup ->> Sup: clear state file
```

## Fleet lifecycle

The fleet transitions through five states. `starting` spawns N detached worker
processes and writes their metadata to state files. `running` is the steady state;
the supervisor monitors PID liveness. `draining` is entered when `stop` or
`restart` sends SIGTERM; workers drain and exit. Once all have exited or been
force-killed and state is cleared, the fleet reaches `stopped`. From `stopped`,
`restart` reads persisted state and reconstructs the fleet in `starting` again.

```mermaid
stateDiagram-v2
    [*] --> starting: start -w N
    starting --> running: N detached workers,<br/>state written
    running --> draining: stop/restart:<br/>SIGTERM
    draining --> stopped: drained or<br/>SIGKILLed;<br/>state cleared
    stopped --> starting: restart reconstructs<br/>from state
    stopped --> [*]
```

## Job lifecycle

Each job in the `classification_jobs` table follows a linear path. The API creates
it in `pending` state. A free worker claims it (moving it to `processing`) and runs
inference off the event loop. On success the job moves to `done`; on inference
error, to `failed`. Both terminal states allow the job to be reaped or archived.

```mermaid
stateDiagram-v2
    [*] --> pending: API creates job
    pending --> processing: free worker picks up
    processing --> done: classified
    processing --> failed: inference error
    done --> [*]
    failed --> [*]
```

## See also

- [supervisor.md](supervisor.md) — CLI reference for `screencrop-supervisorctl`
  (start, stop, restart, status, logs).
- [worker-fleet-tutorial.md](worker-fleet-tutorial.md) — hands-on walkthrough:
  spawning a fleet, consuming the queue, observing metrics.
- [screencrop-pipeline.md](screencrop-pipeline.md) — API endpoints, Prometheus
  metric registry, and configuration reference.
