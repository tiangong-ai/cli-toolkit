import assert from "node:assert/strict";
import { chmod, mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, relative } from "node:path";
import { describe, it } from "node:test";
import { runCli } from "../src/cli.js";
import { registerEvidenceArtifact } from "../src/research/workspace/artifacts.js";
import { lockCapabilities } from "../src/research/workspace/capabilities.js";
import {
  freezeEvidenceContentSnapshot,
  recordArtifactDecomposition,
  registerEvidenceAtom,
} from "../src/research/workspace/content-evidence.js";
import { recordDiscoveryAssessmentBatch } from "../src/research/workspace/discovery.js";
import { bindEvidenceDownload } from "../src/research/workspace/downloads.js";
import { persistBrokerEvidence } from "../src/research/workspace/evidence.js";
import {
  registerBrokerCandidates,
  registerNativeDiscoveryCandidate,
} from "../src/research/workspace/evidence-ledger.js";
import {
  freezeClaimEvidenceGraph,
  freezeInferenceSnapshot,
} from "../src/research/workspace/inference.js";
import { addProjectInput, initializeProject } from "../src/research/workspace/projects.js";
import {
  prepareNativeResearchStage,
  submitNativeResearchStage,
} from "../src/research/workspace/runtime.js";
import {
  canonicalJson,
  sha256Text,
  workspacePaths,
  writeJsonAtomic,
} from "../src/research/workspace/storage.js";
import { scientificDesignInput } from "./helpers/scientific-design.js";
import { initializeResearchWorkspace } from "../src/research/workspace/workspace.js";

describe("read-only literature reuse comparison", () => {
  it("separates normalized DOI identity from exact bytes and preserves paper-specific support and counterevidence", async () => {
    const root = await mkdtemp(join(tmpdir(), "research-literature-reuse-"));
    try {
      await initializeResearchWorkspace(root, undefined);
      await lockCapabilities(root);
      const first = await paper(root, "paper-one", "DOI:10.1234/Example.Work", "support", false, 0);
      const second = await paper(
        root,
        "paper-two",
        "https://doi.org/10.1234/example.work",
        "counterevidence",
        false,
        1,
      );
      const before = await treeBytes(workspacePaths(root).control);
      const originalFetch = globalThis.fetch;
      let networkCalls = 0;
      globalThis.fetch = async () => {
        networkCalls += 1;
        throw new Error("Comparison must remain offline.");
      };
      try {
        const result = await compare(root, "paper-one", "paper-two");
        assert.equal(result.exitCode, 0, result.stderr);
        const view = JSON.parse(result.stdout).literatureReuse;
        assert.equal(view.kind, "tiangong-literature-reuse-comparison");
        assert.deepEqual(view.projectIds, ["paper-one", "paper-two"]);
        assert.equal(view.authority, "advisory-only");
        assert.equal(view.projects[0].scientificApplicability, "not-assessed");
        assert.equal(view.projects[1].authority.state, "authoritative");
        assert.equal(view.projects[0].acquisition.gate.decision, "pass");
        assert.deepEqual(view.projects[0].acquisition.gate.reasons.items, []);
        assert.equal(view.workMatches[0]?.members.items[0]?.recordedDoi, "10.1234/Example.Work");
        assert.equal(view.workMatches.length, 1);
        assert.deepEqual(view.workMatches[0].identity, {
          kind: "doi",
          value: "10.1234/example.work",
        });
        assert.notEqual(first.candidateId, second.candidateId);
        assert.deepEqual(
          view.workMatches[0].members.items.map(
            (member: { candidateId: string }) => member.candidateId,
          ),
          [first.candidateId, second.candidateId],
        );
        assert.notEqual(first.original.sha256, second.original.sha256);
        assert.equal(view.exactFileMatches.length, 1);
        const exact = view.exactFileMatches[0];
        assert.equal(exact.sha256, first.derived.sha256);
        assert.equal(exact.identityConflict, false);
        assert.deepEqual(
          exact.members.items.map(
            (member: { atoms: { items: Array<{ evidenceFunction: string }> } }) =>
              member.atoms.items[0]?.evidenceFunction,
          ),
          ["support", "counterevidence"],
        );
        assert.deepEqual(
          exact.members.items.map(
            (member: {
              atoms: {
                items: Array<{
                  findings: { items: Array<{ findingId: string; claimIds: string[] }> };
                }>;
              };
            }) => member.atoms.items[0]?.findings.items,
          ),
          [
            [
              {
                findingId: "finding:paper-one-finding",
                claimIds: [`design-claim:${first.claimId}`],
              },
            ],
            [
              {
                findingId: "finding:paper-two-finding",
                claimIds: [`design-claim:${second.claimId}`],
              },
            ],
          ],
        );
        assert.ok(
          view.projects.every(
            (project: { content: { status: string }; claimGraph: { status: string } }) =>
              project.content.status === "verified" && project.claimGraph.status === "verified",
          ),
        );
        assert.equal(JSON.stringify(view).includes("Shared exact evidence passage"), false);
        assert.equal(JSON.stringify(view).includes(root), false);
        assert.equal(JSON.stringify(view).includes('"excerpt":'), false);
        assert.equal(networkCalls, 0);
        const replay = await compare(root, "paper-one", "paper-two");
        assert.equal(replay.exitCode, 0, replay.stderr);
        assert.deepEqual(JSON.parse(replay.stdout).literatureReuse, view);
        assert.deepEqual(await treeBytes(workspacePaths(root).control), before);
        const plain = await invoke([
          "research",
          "status",
          "--project",
          "paper-one",
          "--workspace",
          root,
          "--json",
        ]);
        assert.equal(plain.exitCode, 0, plain.stderr);
        assert.equal("literatureReuse" in JSON.parse(plain.stdout), false);
        const otherRoot = join(workspacePaths(root).projects, "paper-two");
        const otherBeforeFork = await treeBytes(otherRoot);
        const forked = await invoke([
          "research",
          "project",
          "fork",
          "paper-one",
          "--to",
          "paper-successor",
          "--resume-through",
          "acquire",
          "--workspace",
          root,
          "--json",
        ]);
        assert.equal(forked.exitCode, 0, forked.stderr);
        const inherited = await compare(root, "paper-successor", "paper-two");
        assert.equal(inherited.exitCode, 0, inherited.stderr);
        const inheritedOutput = JSON.parse(inherited.stdout);
        assert.equal(inheritedOutput.projects[0].usage.tokens, 0);
        assert.equal(inheritedOutput.projects[0].usage.costUsd, 0);
        assert.equal(inheritedOutput.literatureReuse.projects[0].claimGraph.status, "absent");
        const inheritedFile = inheritedOutput.literatureReuse.exactFileMatches.find(
          (file: { sha256: string }) => file.sha256 === first.derived.sha256,
        );
        assert.ok(inheritedFile);
        const successorMember = inheritedFile.members.items.find(
          (member: { projectId: string }) => member.projectId === "paper-successor",
        );
        assert.equal(successorMember.artifactId, first.derived.artifactId);
        assert.deepEqual(
          successorMember.atoms.items[0].locator,
          exact.members.items[0].atoms.items[0].locator,
        );
        assert.deepEqual(successorMember.atoms.items[0].findings.items, []);
        assert.equal(
          inheritedOutput.literatureReuse.projects[0].scientificApplicability,
          "not-assessed",
        );
        assert.deepEqual(await treeBytes(otherRoot), otherBeforeFork);
        assert.equal(networkCalls, 0);
      } finally {
        globalThis.fetch = originalFetch;
      }
      const analysisPath = join(
        workspacePaths(root).projects,
        "paper-one",
        "outputs",
        "analysis.json",
      );
      const analysisBytes = await readFile(analysisPath);
      await writeFile(analysisPath, `${analysisBytes.toString("utf8")}\n`);
      const staleGraph = await compare(root, "paper-one", "paper-two");
      assert.notEqual(staleGraph.exitCode, 0);
      assert.match(staleGraph.stderr, /RESEARCH_CLAIM_EVIDENCE_GRAPH_STALE/);
      await writeFile(analysisPath, analysisBytes);
      await chmod(join(workspacePaths(root).control, first.derived.locator), 0o600);
      await writeFile(
        join(workspacePaths(root).control, first.derived.locator),
        "corrupted bytes\n",
      );
      const corrupted = await compare(root, "paper-one", "paper-two");
      assert.notEqual(corrupted.exitCode, 0);
      assert.match(corrupted.stderr, /RESEARCH_ARTIFACT|RESEARCH_EVIDENCE/);
      assert.equal(corrupted.stdout.includes("exactFileMatches"), false);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it("requires distinct explicit projects and reports unfrozen evidence without inventing eligibility", async () => {
    const root = await mkdtemp(join(tmpdir(), "research-literature-empty-"));
    try {
      await initializeResearchWorkspace(root, undefined);
      await lockCapabilities(root);
      await initializeProject(root, "paper-one", "Compare literature without acquisition.");
      await initializeProject(root, "paper-two", "Compare literature without acquisition.");
      const inputPath = join(root, "pending-input.txt");
      await writeFile(inputPath, "A pending input is outside registered artifact coverage.\n");
      const pendingInput = await addProjectInput(root, "paper-one", inputPath, "primary");
      const before = await treeBytes(workspacePaths(root).control);
      const result = await compare(root, "paper-one", "paper-two");
      assert.equal(result.exitCode, 0, result.stderr);
      const view = JSON.parse(result.stdout).literatureReuse;
      assert.deepEqual(view.workMatches, []);
      assert.deepEqual(view.exactFileMatches, []);
      assert.equal(view.projects[0].scope.projectInputCount, 1);
      assert.deepEqual(view.projects[0].scope.inputsWithoutRegisteredArtifacts.items, [
        pendingInput.id,
      ]);
      assert.ok(
        view.projects.every(
          (project: { acquisition: { status: string }; content: { status: string } }) =>
            project.acquisition.status === "absent" && project.content.status === "absent",
        ),
      );
      assert.ok(view.limitations.length > 0);
      for (const args of [
        ["--compare-project", "paper-two"],
        ["--project", "paper-one", "--compare-project", "paper-one"],
        ["--project", "paper-one", "--compare-project", "paper-two", "--all"],
      ]) {
        const bad = await invoke(["research", "status", ...args, "--workspace", root, "--json"]);
        assert.equal(bad.exitCode, 2, bad.stderr);
        assert.match(bad.stderr, /RESEARCH_LITERATURE_COMPARISON_INVALID/);
      }
      assert.deepEqual(await treeBytes(workspacePaths(root).control), before);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it("keeps conflicting work metadata distinct even when registered bytes match", async () => {
    const root = await mkdtemp(join(tmpdir(), "research-literature-conflict-"));
    try {
      await initializeResearchWorkspace(root, undefined);
      await lockCapabilities(root);
      await paper(root, "paper-one", "10.1234/first", "support");
      await paper(root, "paper-two", "10.1234/second", "counterevidence");
      const projectPath = join(workspacePaths(root).projects, "paper-two", "project.json");
      const partialFork = JSON.parse(await readFile(projectPath, "utf8"));
      partialFork.lineage.kind = "fork";
      partialFork.lineage.derivedFrom = "missing-parent";
      partialFork.lineage.supersedes = "missing-parent";
      await writeFile(projectPath, JSON.stringify(partialFork));
      const result = await compare(root, "paper-one", "paper-two");
      assert.equal(result.exitCode, 0, result.stderr);
      const view = JSON.parse(result.stdout).literatureReuse;
      assert.deepEqual(view.workMatches, []);
      assert.equal(view.exactFileMatches.length, 1);
      assert.equal(view.exactFileMatches[0].identityConflict, true);
      assert.equal(view.projects[1].authority.state, "invalid");
      assert.equal(view.projects[1].scientificApplicability, "not-assessed");
      assert.ok(
        view.exactFileMatches[0].members.items.every(
          (member: { articleVersion: string | null }) => member.articleVersion === null,
        ),
      );
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it("bounds candidate-only output and does not scan an unrelated malformed project", async () => {
    const root = await mkdtemp(join(tmpdir(), "research-literature-bounded-"));
    try {
      await initializeResearchWorkspace(root, undefined);
      await lockCapabilities(root);
      for (const projectId of ["paper-one", "paper-two"]) {
        await initializeProject(root, projectId, "Inspect only selected literature metadata.");
        await prepareNativeResearchStage({
          root,
          projectId,
          stage: "discover",
          hostAgent: "codex",
        });
        for (let index = 0; index < 55; index += 1) {
          await registerNativeDiscoveryCandidate({
            root,
            projectId,
            value: {
              title: "A metadata lead without acquired evidence.",
              url: `https://example.test/${projectId}/${index}`,
              doi: `10.1234/work-${index}`,
            },
          });
        }
        await registerNativeDiscoveryCandidate({
          root,
          projectId,
          value: { title: "Exact URL fallback.", url: "https://example.test/shared-url" },
        });
      }
      for (let index = 0; index < 24; index += 1) {
        await registerNativeDiscoveryCandidate({
          root,
          projectId: "paper-one",
          value: {
            title: "Additional occurrence of the first shared work.",
            url: `https://example.test/extra/${index}`,
            doi: "10.1234/work-0",
          },
        });
      }
      await initializeProject(
        root,
        "unrelated-project",
        "Never scan this unrelated malformed project.",
      );
      await writeFile(
        join(workspacePaths(root).projects, "unrelated-project", "project.json"),
        "invalid unrelated JSON",
      );
      const before = await treeBytes(workspacePaths(root).control);
      const result = await compare(root, "paper-one", "paper-two");
      assert.equal(result.exitCode, 0, result.stderr);
      const view = JSON.parse(result.stdout).literatureReuse;
      assert.equal(view.workMatches.length, view.limits.groupsPerKind);
      assert.equal(view.omitted.workGroups, 6);
      const crowded = view.workMatches.find(
        (group: { identity: { value: string } }) => group.identity.value === "10.1234/work-0",
      );
      assert.equal(crowded.members.items.length, view.limits.itemsPerList);
      assert.equal(crowded.members.omitted, 6);
      assert.deepEqual(
        [
          ...new Set(
            crowded.members.items.map((member: { projectId: string }) => member.projectId),
          ),
        ].sort(),
        ["paper-one", "paper-two"],
      );
      assert.ok(Buffer.byteLength(JSON.stringify(view)) <= view.limits.maxComparisonBytes);
      assert.ok(
        view.projects.every(
          (project: { content: { status: string }; claimGraph: { status: string } }) =>
            project.content.status === "absent" && project.claimGraph.status === "absent",
        ),
      );
      assert.deepEqual(await treeBytes(workspacePaths(root).control), before);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it("folds only DOI ASCII case and preserves Unicode case and suffix punctuation", async () => {
    const root = await mkdtemp(join(tmpdir(), "research-literature-doi-"));
    try {
      await initializeResearchWorkspace(root, undefined);
      await lockCapabilities(root);
      const dois = {
        "paper-one": ["10.1234/ASCII", "10.1234/Á", "10.1234/suffix."],
        "paper-two": ["10.1234/ascii", "10.1234/á", "10.1234/suffix"],
      };
      for (const [projectId, values] of Object.entries(dois)) {
        await initializeProject(root, projectId, "Compare conservative DOI metadata equivalence.");
        await prepareNativeResearchStage({
          root,
          projectId,
          stage: "discover",
          hostAgent: "codex",
        });
        for (const [index, doi] of values.entries()) {
          await registerNativeDiscoveryCandidate({
            root,
            projectId,
            value: {
              title: "DOI equivalence fixture.",
              url: `https://example.test/${projectId}/${index}`,
              doi,
            },
          });
        }
      }
      const result = await compare(root, "paper-one", "paper-two");
      assert.equal(result.exitCode, 0, result.stderr);
      const view = JSON.parse(result.stdout).literatureReuse;
      assert.deepEqual(
        view.workMatches.map((group: { identity: { value: string } }) => group.identity.value),
        ["10.1234/ascii"],
      );
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it("distinguishes verified bindings from a stopped scientific gate", async () => {
    const root = await mkdtemp(join(tmpdir(), "research-literature-stopped-"));
    try {
      await initializeResearchWorkspace(root, undefined);
      await lockCapabilities(root);
      await paper(root, "paper-one", "10.1234/one", "counterevidence", true);
      await initializeProject(root, "paper-two", "Inspect one honestly stopped evidence chain.");
      const result = await compare(root, "paper-one", "paper-two");
      assert.equal(result.exitCode, 0, result.stderr);
      const project = JSON.parse(result.stdout).literatureReuse.projects[0];
      assert.equal(project.acquisition.status, "verified");
      assert.equal(project.acquisition.gate.decision, "stop");
      assert.deepEqual(project.acquisition.gate.reasons.items, ["A named material gap remains."]);
      assert.ok(project.limitations.items.includes("A named material gap remains."));
      assert.equal(project.scientificApplicability, "not-assessed");
      assert.equal(project.claimGraph.status, "absent");
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it("refuses changed frozen content instead of treating it as reusable", async () => {
    const root = await mkdtemp(join(tmpdir(), "research-literature-drift-"));
    try {
      await initializeResearchWorkspace(root, undefined);
      await lockCapabilities(root);
      await paper(root, "paper-one", "10.1234/one", "support");
      await initializeProject(root, "paper-two", "Inspect selected existing evidence.");
      const path = join(
        workspacePaths(root).projects,
        "paper-one",
        "outputs",
        "content-snapshot.json",
      );
      await chmod(path, 0o600);
      const snapshot = JSON.parse(await readFile(path, "utf8"));
      snapshot.atoms[0].statement = "Changed after freeze.";
      await writeFile(path, JSON.stringify(snapshot));
      const result = await compare(root, "paper-one", "paper-two");
      assert.notEqual(result.exitCode, 0);
      assert.match(result.stderr, /RESEARCH_EVIDENCE_(?:CONTENT_SNAPSHOT|ATOM_STORE)_INVALID/);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });
});

async function paper(
  root: string,
  projectId: string,
  doi: string,
  evidenceFunction: string,
  stopped = false,
  designClaimIndex?: number,
) {
  await initializeProject(
    root,
    projectId,
    "Compare evidence while preserving paper-specific applicability.",
  );
  const discover = await prepareNativeResearchStage({
    root,
    projectId,
    stage: "discover",
    hostAgent: "codex",
  });
  const url = `https://example.test/${projectId}`;
  const contextBytes = Buffer.from(JSON.stringify([{ title: "Shared literature", url, doi }]));
  const receipt = await persistBrokerEvidence(
    root,
    {
      attemptId: `${projectId}-receipt`,
      projectId,
      capabilityId: "method.fixture",
      credentialId: null,
      status: 200,
      contentType: "application/json",
      sourceSha256: "a".repeat(64),
      contextItems: 1,
      contextOffset: 0,
      contextTotalItems: 1,
      contextNextOffset: null,
      contextTruncated: false,
      redactions: 0,
      retrievedAt: "2026-10-20T00:00:00.000Z",
      cacheHit: false,
    },
    contextBytes,
    contextBytes,
  );
  const [candidate] = await registerBrokerCandidates({ root, projectId, receipt, contextBytes });
  assert.ok(candidate);
  await recordDiscoveryAssessmentBatch({
    root,
    projectId,
    value: {
      schemaVersion: 1,
      assessments: [
        {
          decision: "admit",
          candidateId: candidate.id,
          sourceId: `${projectId}-source`,
          sourceType: "primary",
          relevance:
            evidenceFunction === "counterevidence"
              ? "Contrary comparison."
              : "Supporting comparison.",
          quality: { level: "primary", rationale: "Offline exact fixture." },
          applicability: "Paper-specific applicability.",
          coverageDimensions: ["research-question"],
          limitations: [],
        },
      ],
    },
  });
  const output = join(root, `${projectId}-submit.json`);
  await writeFile(
    output,
    JSON.stringify({
      schemaVersion: 2,
      limitations: [],
      dimensionJudgments: [{ id: "research-question", status: "covered" }],
      gaps: [],
    }),
  );
  await submitNativeResearchStage({
    root,
    projectId,
    sessionId: discover.sessionId,
    outputPath: output,
    confirmedModel: discover.expectedModel,
  });
  const acquire = await prepareNativeResearchStage({
    root,
    projectId,
    stage: "acquire",
    hostAgent: "codex",
  });
  const source = join(root, `${projectId}.txt`);
  await writeFile(source, `${projectId} version\nShared exact evidence passage.\n`);
  const download = await bindEvidenceDownload({
    root,
    projectId,
    candidateId: candidate.id,
    value: {
      schemaVersion: 1,
      backend: "native-browser",
      status: "completed",
      path: source,
      downloadUrl: url,
      suggestedFilename: `${projectId}.txt`,
      downloadIdentifier: `${projectId}-download`,
    },
  });
  assert.equal(download.status, "completed");
  const original = await registerEvidenceArtifact({
    root,
    projectId,
    candidateId: candidate.id,
    path: source,
    articleVersion: projectId,
    downloadBindingId: download.status === "completed" ? download.binding.bindingId : "",
  });
  const derivedPath = join(root, `${projectId}-derived.txt`);
  await writeFile(derivedPath, "Shared exact evidence passage.\n");
  const derived = await registerEvidenceArtifact({
    root,
    projectId,
    candidateId: candidate.id,
    path: derivedPath,
    derivedFromArtifactId: original.artifactId,
  });
  await writeFile(
    output,
    JSON.stringify({
      schemaVersion: 1,
      decisions: [
        {
          sourceId: `${projectId}-source`,
          candidateId: candidate.id,
          artifactIds: [original.artifactId, derived.artifactId],
          status: "accepted",
          rationale: "Exact fixture artifacts.",
          limitations: [],
        },
      ],
      limitations: [],
      gaps: stopped ? ["A named material gap remains."] : [],
    }),
  );
  await submitNativeResearchStage({
    root,
    projectId,
    sessionId: acquire.sessionId,
    outputPath: output,
    confirmedModel: acquire.expectedModel,
  });
  await recordArtifactDecomposition({
    root,
    projectId,
    value: {
      schemaVersion: 1,
      sourceArtifactId: original.artifactId,
      status: "complete",
      parser: { id: "fixture.lines", version: "1" },
      outputArtifactIds: [derived.artifactId],
      contentClasses: ["fulltext"],
      limitations: [],
    },
  });
  const atom = await registerEvidenceAtom({
    root,
    projectId,
    value: {
      schemaVersion: 1,
      atomId: `${projectId}-atom`,
      sourceId: `${projectId}-source`,
      candidateId: candidate.id,
      artifactId: derived.artifactId,
      locator: { kind: "line-range", startLine: 1, endLine: 1 },
      statement: "A paper-specific interpretation.",
      evidenceRoleIds: [],
      coverageDimensionIds: ["research-question"],
      evidenceFunction,
      scope: "The selected paper's question.",
      limitations: [],
    },
  });
  await freezeEvidenceContentSnapshot(root, projectId);
  if (stopped) return { candidateId: candidate.id, original, derived, claimId: null };
  const inference = await freezeInferenceSnapshot(root, projectId);
  let claimId: string | null = null;
  if (designClaimIndex !== undefined) {
    const design = await scientificDesignInput(root, `${projectId}-claims`);
    const claim = design.design.contract.claims[designClaimIndex];
    assert.ok(claim);
    claimId = claim.id;
    // Synthetic, hash-bound graph fixture, following the publication workflow
    // fixtures. The design is validated; this does not simulate review approval.
    const { snapshotSha256: _previousHash, ...core } = inference;
    const bound = { ...core, claims: [claim] };
    const snapshot = { ...bound, snapshotSha256: sha256Text(canonicalJson(bound)) };
    const projectRoot = join(workspacePaths(root).projects, projectId);
    await writeJsonAtomic(
      join(projectRoot, "evidence", "inference-snapshots", `${snapshot.snapshotSha256}.json`),
      snapshot,
    );
    await writeJsonAtomic(join(projectRoot, "outputs", "inference-snapshot.json"), snapshot);
  }
  const analysis = {
    findings: [
      {
        id: `${projectId}-finding`,
        statement: "A paper-specific conclusion.",
        evidenceAtomIds: [atom.atomId],
        claimIds: claimId ? [claimId] : [],
      },
    ],
    analysisRun: { id: `${projectId}-run` },
  };
  await writeFile(
    join(workspacePaths(root).projects, projectId, "outputs", "analysis.json"),
    JSON.stringify(analysis),
  );
  await freezeClaimEvidenceGraph(root, projectId, analysis);
  return { candidateId: candidate.id, original, derived, claimId };
}
async function compare(root: string, first: string, second: string) {
  return invoke([
    "research",
    "status",
    "--project",
    first,
    "--compare-project",
    second,
    "--workspace",
    root,
    "--json",
  ]);
}
async function invoke(argv: string[]) {
  let stdout = "";
  let stderr = "";
  const exitCode = await runCli(argv, {
    env: {},
    stdout: { write: (chunk) => ((stdout += chunk), true) },
    stderr: { write: (chunk) => ((stderr += chunk), true) },
  });
  return { exitCode, stdout, stderr };
}
async function treeBytes(root: string): Promise<Array<[string, string]>> {
  const result: Array<[string, string]> = [];
  async function walk(directory: string) {
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      const path = join(directory, entry.name);
      if (entry.isDirectory()) await walk(path);
      else if (entry.isFile())
        result.push([relative(root, path), (await readFile(path)).toString("base64")]);
    }
  }
  await walk(root);
  return result.sort(([left], [right]) => left.localeCompare(right));
}
