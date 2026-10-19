import { join } from "node:path";

import { CliError } from "../../errors.js";
import { loadCurrentEvidenceSnapshot } from "./acquisition.js";
import { loadEvidenceArtifactRecords, type EvidenceArtifactRecord } from "./artifacts.js";
import { loadCurrentEvidenceContentSnapshot, type EvidenceAtomRecord } from "./content-evidence.js";
import {
  evidenceLedgerPath,
  listEvidenceCandidates,
  type EvidenceCandidate,
} from "./evidence-ledger.js";
import { loadCurrentClaimEvidenceGraph, type ClaimEvidenceGraph } from "./inference.js";
import { readVerifiedJournal } from "./journal.js";
import { loadProject } from "./projects.js";
import { projectAuthority, type ProjectAuthorityIndex } from "./project-authority.js";
import { pathExists, workspacePaths } from "./storage.js";

const MAX_GROUPS = 50;
const MAX_ITEMS = 20;
const MAX_COMPARISON_BYTES = 256 * 1024;

type WorkIdentity = { kind: "doi" | "url"; value: string };

/** A derived view only: none of these keys replaces a persisted candidate ID. */
function workIdentity(candidate: EvidenceCandidate): WorkIdentity | null {
  const doi = candidate.doi
    ?.trim()
    .replace(/^https?:\/\/(?:dx\.)?doi\.org\//i, "")
    .replace(/^doi:\s*/i, "")
    // DOI Handbook section 3.3.4 defines ASCII case equivalence only.
    // https://www.doi.org/doi-handbook/html/
    .replace(/[A-Z]/g, (letter) => letter.toLowerCase());
  if (doi && /^10\.\d{4,9}\/\S+$/.test(doi)) return { kind: "doi", value: doi };
  return candidate.url ? { kind: "url", value: candidate.url } : null;
}

function bounded<T>(items: T[]) {
  return { items: items.slice(0, MAX_ITEMS), omitted: Math.max(0, items.length - MAX_ITEMS) };
}

/** Keep both selected projects visible even when one has many associations. */
function boundedMembers<T extends { projectId: string }>(members: T[]) {
  const projects = [...new Set(members.map((member) => member.projectId))];
  const columns = projects.map((projectId) =>
    members.filter((member) => member.projectId === projectId),
  );
  const items: T[] = [];
  for (let index = 0; items.length < MAX_ITEMS && index < MAX_ITEMS; index += 1) {
    for (const column of columns) {
      if (column[index] && items.length < MAX_ITEMS) items.push(column[index]!);
    }
  }
  return { items, omitted: members.length - items.length };
}

function findingsForAtom(graph: ClaimEvidenceGraph | null, atomId: string) {
  if (!graph) return bounded([] as Array<{ findingId: string; claimIds: string[] }>);
  return bounded(
    graph.edges
      .filter((edge) => edge.type === "finding-supported-by-atom" && edge.to === `atom:${atomId}`)
      .map((edge) => ({
        findingId: edge.from,
        claimIds: graph.edges
          .filter(
            (claim) => claim.type === "finding-addresses-design-claim" && claim.from === edge.from,
          )
          .map((claim) => claim.to)
          .sort(),
      }))
      .sort((left, right) => left.findingId.localeCompare(right.findingId)),
  );
}

async function inspectProject(
  root: string,
  projectId: string,
  authorityIndex: ProjectAuthorityIndex,
) {
  const project = await loadProject(root, projectId);
  const ledgerPath = evidenceLedgerPath(root, projectId);
  const events = await readVerifiedJournal(ledgerPath);
  const candidates = await listEvidenceCandidates(root, projectId);
  const artifacts = await loadEvidenceArtifactRecords(root, projectId);
  const outputs = join(workspacePaths(root).projects, projectId, "outputs");
  const acquisition = (await pathExists(join(outputs, "evidence-snapshot.json")))
    ? await loadCurrentEvidenceSnapshot(root, projectId)
    : null;
  const content = (await pathExists(join(outputs, "content-snapshot.json")))
    ? await loadCurrentEvidenceContentSnapshot(root, projectId)
    : null;
  const graph = (await pathExists(join(outputs, "claim-evidence-graph.json")))
    ? await loadCurrentClaimEvidenceGraph(root, projectId)
    : null;
  const sourceCandidates = new Map<string, string>();
  for (const event of events) {
    if (
      event.type === "candidate.admitted" &&
      typeof event.payload.sourceId === "string" &&
      typeof event.payload.candidateId === "string"
    ) {
      sourceCandidates.set(event.payload.sourceId, event.payload.candidateId);
    }
  }
  const currentArtifactIds = new Set(acquisition?.artifacts.map((artifact) => artifact.artifactId));
  const atomView = (atom: EvidenceAtomRecord) => ({
    atomId: atom.atomId,
    atomSha256: atom.atomSha256,
    sourceId: atom.sourceId,
    artifactId: atom.artifactId,
    artifactSha256: atom.artifactSha256,
    locator: atom.locator,
    evidenceFunction: atom.evidenceFunction,
    evidenceRoleIds: bounded(atom.evidenceRoleIds),
    coverageDimensionIds: bounded(atom.coverageDimensionIds),
    findings: findingsForAtom(graph, atom.atomId),
  });
  const artifactView = (artifact: EvidenceArtifactRecord) => ({
    projectId,
    candidateId: artifact.candidateId,
    artifactId: artifact.artifactId,
    sha256: artifact.sha256,
    bytes: artifact.bytes,
    mediaType: artifact.mediaType,
    articleVersion: artifact.articleVersion,
    derivedFromArtifactId: artifact.derivedFromArtifactId,
    downloadBindingSha256: artifact.downloadBinding?.bindingSha256 ?? null,
    license: artifact.license,
    inCurrentAcquisition: currentArtifactIds.has(artifact.artifactId),
    atoms: bounded(
      (content?.atoms ?? [])
        .filter((atom) => atom.artifactId === artifact.artifactId)
        .map(atomView),
    ),
  });
  const members = candidates.map((candidate) => ({
    projectId,
    candidateId: candidate.id,
    identity: workIdentity(candidate),
    recordedDoi: candidate.doi,
    provenance: bounded(
      candidate.occurrences.map((origin) => ({
        kind: origin.kind,
        receiptId: origin.receiptId,
        inputId: origin.inputId,
        capabilityId: origin.capabilityId,
        jsonPointer: origin.jsonPointer,
        retrievedAt: origin.retrievedAt,
      })),
    ),
    sources: bounded(
      (acquisition?.sources ?? [])
        .filter((source) => sourceCandidates.get(String(source.id)) === candidate.id)
        .map((source) => ({
          sourceId: String(source.id),
          acquisitionStatus:
            typeof source.acquisitionStatus === "string" ? source.acquisitionStatus : null,
          fullTextAvailable: source.fullTextAvailable === true,
        })),
    ),
    artifacts: bounded(
      artifacts.filter((artifact) => artifact.candidateId === candidate.id).map(artifactView),
    ),
  }));
  const candidateIds = new Set(candidates.map((candidate) => candidate.id));
  if (artifacts.some((artifact) => !candidateIds.has(artifact.candidateId))) {
    throw new CliError("Registered artifact refers to an unknown evidence candidate.", {
      code: "RESEARCH_LITERATURE_COMPARISON_INVALID",
      exitCode: 3,
    });
  }
  const files = artifacts.map((artifact) => ({
    ...artifactView(artifact),
    identity: workIdentity(candidates.find((candidate) => candidate.id === artifact.candidateId)!),
    recordedDoi: candidates.find((candidate) => candidate.id === artifact.candidateId)!.doi,
  }));
  if ((await readVerifiedJournal(ledgerPath)).at(-1)?.hash !== events.at(-1)?.hash) {
    throw new CliError("Evidence changed during literature comparison; inspect again.", {
      code: "RESEARCH_LITERATURE_COMPARISON_CHANGED",
      exitCode: 3,
    });
  }
  return {
    state: {
      projectId,
      authority: projectAuthority(project, authorityIndex),
      scientificApplicability: "not-assessed",
      scope: {
        candidateCount: candidates.length,
        registeredArtifactCount: artifacts.length,
        candidatesWithoutWorkIdentity: candidates.filter(
          (candidate) => workIdentity(candidate) === null,
        ).length,
        projectInputCount: project.inputs.length,
        inputsWithoutRegisteredArtifacts: bounded(
          project.inputs
            .filter(
              (input) =>
                !candidates.some(
                  (candidate) =>
                    candidate.occurrences.some((origin) => origin.inputId === input.id) &&
                    artifacts.some((artifact) => artifact.candidateId === candidate.id),
                ),
            )
            .map((input) => input.id),
        ),
      },
      limitations: bounded([
        ...(acquisition?.limitations ?? []),
        ...(acquisition?.gaps ?? []),
        ...(content?.gate.reasons ?? []),
      ]),
      acquisition: acquisition
        ? {
            status: "verified",
            snapshotSha256: acquisition.snapshotSha256,
            gate: {
              decision: acquisition.inferenceGate.decision,
              reasons: bounded(acquisition.inferenceGate.reasons),
            },
          }
        : { status: "absent" },
      content: content
        ? {
            status: "verified",
            snapshotSha256: content.snapshotSha256,
            gate: { decision: content.gate.decision, reasons: bounded(content.gate.reasons) },
          }
        : { status: "absent" },
      claimGraph: graph
        ? { status: "verified", graphSha256: graph.graphSha256 }
        : { status: "absent" },
    },
    members,
    files,
  };
}

function sharedGroups<T extends { projectId: string }>(
  items: T[],
  key: (item: T) => string | null,
) {
  const groups = new Map<string, T[]>();
  for (const item of items) {
    const identity = key(item);
    if (identity !== null) groups.set(identity, [...(groups.get(identity) ?? []), item]);
  }
  return [...groups.entries()]
    .filter(([, members]) => new Set(members.map((member) => member.projectId)).size === 2)
    .sort(([left], [right]) => left.localeCompare(right));
}

/** Offline comparison of exactly two projects; it never admits or copies evidence. */
export async function inspectLiteratureReuse(
  root: string,
  firstProjectId: string,
  secondProjectId: string,
  authorityIndex: ProjectAuthorityIndex,
) {
  const projects = [
    await inspectProject(root, firstProjectId, authorityIndex),
    await inspectProject(root, secondProjectId, authorityIndex),
  ];
  const workGroups = sharedGroups(
    projects.flatMap((project) => project.members),
    (member) => (member.identity ? `${member.identity.kind}:${member.identity.value}` : null),
  );
  const fileGroups = sharedGroups(
    projects.flatMap((project) => project.files),
    (file) => file.sha256,
  );
  const workMatches: Array<{
    identity: WorkIdentity;
    members: ReturnType<typeof bounded<(typeof projects)[number]["members"][number]>>;
  }> = [];
  const exactFileMatches: Array<{
    sha256: string;
    identityConflict: boolean;
    members: ReturnType<typeof bounded<(typeof projects)[number]["files"][number]>>;
  }> = [];
  // Reserve space for the envelope and explicit omissions. A large group may be
  // omitted rather than silently dropping evidence inside its members.
  let remainingBytes = MAX_COMPARISON_BYTES - 16 * 1024;
  function include(value: unknown) {
    const bytes = Buffer.byteLength(JSON.stringify(value));
    if (bytes > remainingBytes) return false;
    remainingBytes -= bytes;
    return true;
  }
  for (const [, members] of workGroups) {
    const group = { identity: members[0]!.identity!, members: boundedMembers(members) };
    if (workMatches.length < MAX_GROUPS && include(group)) workMatches.push(group);
  }
  for (const [sha256, members] of fileGroups) {
    const dois = new Set(
      members
        .filter((member) => member.identity?.kind === "doi")
        .map((member) => member.identity!.value),
    );
    const group = { sha256, identityConflict: dois.size > 1, members: boundedMembers(members) };
    if (exactFileMatches.length < MAX_GROUPS && include(group)) exactFileMatches.push(group);
  }
  const comparison = {
    schemaVersion: 1,
    kind: "tiangong-literature-reuse-comparison",
    authority: "advisory-only",
    projectIds: [firstProjectId, secondProjectId],
    projects: projects.map((project) => project.state),
    limits: {
      groupsPerKind: MAX_GROUPS,
      itemsPerList: MAX_ITEMS,
      maxComparisonBytes: MAX_COMPARISON_BYTES,
    },
    workMatches,
    exactFileMatches,
    omitted: {
      workGroups: workGroups.length - workMatches.length,
      fileGroups: fileGroups.length - exactFileMatches.length,
    },
    limitations: [
      "Work identity and byte matches do not establish current relevance, lawful reuse, substantive reading, admission, or scientific approval. Historical IDs are unchanged.",
      "Missing acquisition/content/claim graphs remain absent; registered files outside the current acquisition are marked explicitly. Only frozen verified atoms and finding-to-claim bindings are projected.",
      "Different hashes remain different representations even for one DOI; articleVersion is nullable user-declared metadata, not verified same-version identity. Conflicting DOI metadata on identical bytes requires investigation.",
      "Only registered artifact hashes are compared, not an exhaustive input corpus; unrepresented input IDs and candidates without work identity are counted. This bounded metadata view omits source excerpts and filesystem locator fields; declared metadata and limitations remain unmodified. List omissions and group omissions are explicit; use the existing exact artifact read channel for further appraisal.",
      "No actual token, billing, reading-time, or savings measurement is inferred by this comparison.",
    ],
  };
  // Gate reasons and limitations are data too: enforce the byte ceiling on the
  // complete envelope, not only the match arrays.
  while (Buffer.byteLength(JSON.stringify(comparison)) > MAX_COMPARISON_BYTES) {
    if (comparison.exactFileMatches.pop()) comparison.omitted.fileGroups += 1;
    else if (comparison.workMatches.pop()) comparison.omitted.workGroups += 1;
    else
      throw new CliError(
        "Comparison metadata exceeds its byte limit; inspect each project separately.",
        {
          code: "RESEARCH_LITERATURE_COMPARISON_TOO_LARGE",
          exitCode: 3,
        },
      );
  }
  return comparison;
}
