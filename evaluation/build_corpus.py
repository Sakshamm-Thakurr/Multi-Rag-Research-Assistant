"""
Builds evaluation/corpus/quillbase_handbook.pdf (developer tool; needs `pip install reportlab`).
Quillbase is a FICTIONAL database invented for this project: its facts exist only in this document,
so the language model cannot answer from memory, and there is no third-party copyright.
"""
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

SECTIONS = [
    ("1. Overview", [
        "Quillbase is a distributed key-value store written in Rust. This handbook describes release 4.2 and is written for operators who run production clusters. Quillbase is a fictional product invented to demonstrate retrieval evaluation.",
        "A cluster is made of nodes that share membership information through a gossip protocol and keep data consistent with a replicated log. Any node can accept a client request and forward it to the right owner.",
    ]),
    ("2. Network ports and TLS", [
        "Clients connect to a node on TCP port 7421 by default. The administration interface listens on port 7422 and must never be exposed to the public internet. Nodes exchange gossip messages with each other on port 7430.",
        "Every connection that does not come from the loopback interface must use TLS 1.3. Plain-text connections from other hosts are refused, and the refusal is written to the audit log.",
    ]),
    ("3. Storage engine", [
        "Quillbase writes every change to a write-ahead log before it is applied to the data files. The log is split into segments of 64 MiB each, and old segments are archived once every replica has applied them.",
        "By default the log is flushed to disk every 5 ms using group commit, which trades a small delay for much higher throughput. Data files use a block size of 16 KiB. Background compaction starts when 4 overlapping tables exist at the same level.",
    ]),
    ("4. Replication", [
        "Each key is stored with a replication factor of 3 by default, so three different nodes hold a copy of it. A write is acknowledged to the client once 2 of 3 replicas have confirmed it, which is called a quorum write.",
        "Read repair is enabled by default: when a read notices that two replicas disagree, the stale copy is fixed in the background. Operators should raise an alert when cross-region replication lag stays above 800 ms.",
    ]),
    ("5. Snapshots and restore", [
        "Snapshots run every night at 02:30 UTC and are kept for 14 days. Each snapshot has an identifier that starts with the prefix qb-snap- followed by the date and a short hash.",
        "To restore a node, run qbctl restore with the snapshot identifier. Restoring replaces all local data on the node, so drain the node from the load balancer first and let replication refill it afterwards.",
    ]),
    ("6. Memory", [
        "The block cache uses 25% of system memory by default. The setting cache_ratio can raise it, but the hard limit is 70% of memory and larger values are rejected at startup.",
        "Leave the rest of the memory free for the operating system page cache. Running the cache close to the hard limit makes compaction slower and increases the risk of the process being stopped by the kernel.",
    ]),
    ("7. Security and access", [
        "API tokens stay valid for 12 hours and must then be renewed by the client. Three roles exist: reader, writer and operator. Only the operator role may change cluster settings or trigger a restore.",
        "The audit log records every administrative action together with the token that performed it. Audit entries are retained for 90 days and can be exported as JSON lines.",
    ]),
    ("8. Upgrades", [
        "Clusters are upgraded with a rolling upgrade that touches one node at a time. Wait at least 5 minutes between nodes so that replication lag can return to normal before the next node restarts.",
        "Downgrading across a major version is not supported. Always take a snapshot before an upgrade and check the release notes for changes to the on-disk format.",
    ]),
    ("9. Monitoring", [
        "The metric qb_commit_latency_p99 tracks the 99th percentile commit latency. The default alert fires when it stays above 25 ms for ten minutes.",
        "The bundled dashboard is called Heronwatch. It shows replication lag, cache hit ratio and the compaction backlog for every node in one view.",
    ]),
    ("10. Error codes", [
        "QB-4091 means that quorum was lost because too many replicas are unreachable. QB-5003 means that the data disk is full and writes are rejected until space is freed. QB-2210 means that the API token has expired and must be renewed.",
    ]),
    ("11. Limits", [
        "The maximum key size is 1 KiB and the maximum value size is 8 MiB. A single batch may contain at most 500 keys. A cluster supports up to 256 namespaces.",
    ]),
    ("12. Support", [
        "The support desk is open on weekdays from 08:00 to 18:00 UTC. A Severity 1 incident, which means a full outage, receives a first response within 30 minutes. Severity 2 incidents receive a response within 4 hours.",
    ]),
]

# Appendices: deliberately confusable reference lines (similar wording, different numbers) so that
# retrieval has to discriminate. None of these values repeats a value asked about in golden.json.
SETTINGS = [
    ("gossip_interval", "Time between gossip rounds", "200 ms"), ("heartbeat_timeout", "Time before a peer is suspected", "1500 ms"),
    ("max_connections", "Client connections per node", "4096"), ("idle_timeout", "Idle client connection limit", "300 seconds"),
    ("read_timeout", "Deadline for a read request", "2000 ms"), ("write_timeout", "Deadline for a write request", "3000 ms"),
    ("cache_shards", "Number of block cache shards", "16"), ("bloom_bits_per_key", "Bloom filter size per key", "10 bits"),
    ("query_log_retention", "How long query logs are kept", "7 days"), ("metrics_retention", "How long metrics are kept", "30 days"),
    ("wal_archive_retention", "How long archived log segments are kept", "3 days"), ("max_replication_streams", "Parallel replication streams", "8"),
    ("rebalance_rate", "Data moved per second during a rebalance", "48 MiB"), ("token_renewal_window", "How early a token may be renewed", "15 minutes"),
    ("login_attempt_limit", "Failed logins before lockout", "6"), ("lockout_duration", "Length of an account lockout", "20 minutes"),
    ("tls_cert_reload", "Interval for reloading certificates", "60 seconds"), ("audit_flush", "Interval for flushing the audit log", "10 seconds"),
    ("compaction_threads", "Background compaction workers", "2"), ("snapshot_compression", "Snapshot compression level", "zstd level 3"),
    ("batch_timeout", "Deadline for a batch request", "4000 ms"), ("gossip_fanout", "Peers contacted per gossip round", "3 peers"),
    ("hint_ttl", "How long hinted handoffs are stored", "2 hours"), ("scan_page_size", "Rows returned per scan page", "1000"),
    ("ttl_sweep_interval", "How often expired keys are removed", "90 seconds"), ("checkpoint_interval", "Time between checkpoints", "10 minutes"),
    ("max_open_files", "File descriptors per process", "65536"), ("slow_query_threshold", "Latency that marks a slow query", "750 ms"),
    ("quota_check_interval", "How often namespace quotas are checked", "45 seconds"), ("log_rotation_size", "Size of an application log file", "200 MiB"),
    ("dns_refresh", "Interval for refreshing peer addresses", "120 seconds"), ("election_timeout", "Time before a leader election", "3500 ms"),
    ("range_split_size", "Size at which a range is split", "512 MiB"), ("range_merge_size", "Size below which ranges are merged", "64 KiB"),
    ("client_retry_limit", "Retries performed by the client library", "3"), ("backoff_base", "Starting delay for retry backoff", "40 ms"),
]
CODES = [
    ("QB-1001", "configuration file could not be parsed"), ("QB-1007", "unknown setting name in the configuration"),
    ("QB-1013", "certificate file is missing or unreadable"), ("QB-2101", "namespace does not exist"),
    ("QB-2102", "namespace name is not valid"), ("QB-2144", "request is not allowed for this role"),
    ("QB-2301", "key is larger than the key size limit"), ("QB-2302", "value is larger than the value size limit"),
    ("QB-3012", "node is draining and refuses new writes"), ("QB-3040", "range is being split, retry shortly"),
    ("QB-3077", "hinted handoff queue is full"), ("QB-4010", "replica did not answer within the write timeout"),
    ("QB-4022", "read repair was skipped under load"), ("QB-4108", "leader election is in progress"),
    ("QB-4150", "clock skew between nodes is too large"), ("QB-5011", "compaction backlog is above the safe level"),
    ("QB-5020", "block cache could not be allocated"), ("QB-5044", "log segment checksum mismatch"),
    ("QB-6001", "snapshot upload to object storage failed"), ("QB-6014", "snapshot identifier was not found"),
    ("QB-6020", "snapshot belongs to a different major version"), ("QB-7003", "audit log export is already running"),
    ("QB-7010", "operator token required for this command"), ("QB-8002", "client library version is too old"),
]


def build(path: Path) -> None:
    st = getSampleStyleSheet()
    story = []
    for i, (title, paras) in enumerate(SECTIONS):
        story.append(Paragraph(title, st["Heading2"]))
        story += [Paragraph(p, st["BodyText"]) for p in paras]
        story.append(Spacer(1, 10))
        if i in (3, 7):
            story.append(PageBreak())
    story.append(PageBreak())
    story.append(Paragraph("Appendix A. Configuration reference", st["Heading2"]))
    for name, desc, default in SETTINGS:
        story.append(Paragraph(f"{name}: {desc}. Default value: {default}.", st["BodyText"]))
    story.append(PageBreak())
    story.append(Paragraph("Appendix B. Error code catalog", st["Heading2"]))
    for code, meaning in CODES:
        story.append(Paragraph(f"{code} means that the {meaning}.", st["BodyText"]))
    SimpleDocTemplate(str(path), pagesize=A4, title="Quillbase Operations Handbook").build(story)


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "corpus" / "quillbase_handbook.pdf"
    build(out)
    print("wrote", out)
