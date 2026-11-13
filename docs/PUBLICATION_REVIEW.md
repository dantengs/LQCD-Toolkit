# Publication review

Date: 2026-11-14. **The reviewed tree satisfies the new-code-only publication
scope.** No remaining privacy or license-scope issue was identified in this
bounded, read-only review.

Before adding this report, the tree contained 63 release files: 61 text files and
two synthetic numeric binaries. The independent review found no historical
source archive, old project tree, notebook, compiled bytecode, database, private
backup, or symbolic link. The compatibility and diagnostics modules are new
implementation code; supporting an existing format does not distribute its old
writer. No personal machine paths, private endpoints, contact identities, or
literal credentials were found by the checked patterns and Python-string review.
Absolute path literals were confined to explicitly synthetic test inputs.

All 34 Python files parsed successfully. Configuration templates use generic,
incomplete inputs with their template guards enabled. All four parser-fixture
files match the synthetic provenance manifest, including the two allowed
binaries. The public-tree checker passed with zero problems. Independent link
checks resolved all 56 local Markdown targets, including 36 in the two READMEs.
No source file was changed or executed as a production workflow.

The [MIT license](../LICENSE) and [dependency boundaries](../THIRD_PARTY.md)
consistently cover the distributed new first-party implementation, tests,
synthetic fixtures, templates, and documentation. External numerical libraries
and the separately supplied solver retain their own licenses; no vendor source
or historical source bundle is included.

This conclusion concerns the reviewed local files, not every possible future
change or an independent determination of copyright ownership. Git initialization,
commit parentage, repository creation, remote visibility, and server-side history
were not operated on or verified here. The review makes no claim that any prior
repository history or cached commit has been removed.
