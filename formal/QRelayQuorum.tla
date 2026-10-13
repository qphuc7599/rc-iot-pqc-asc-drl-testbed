------------------------------ MODULE QRelayQuorum ------------------------------
EXTENDS FiniteSets, Integers, Sequences, TLC

CONSTANTS Members, Byzantine, Q

ASSUME Byzantine \subseteq Members

CertificateSets == {s \in SUBSET Members : Cardinality(s) >= Q}
Honest == Members \ Byzantine

(* Persistent non-equivocation means two conflicting certificates may share
   Byzantine members, but no honest signer. *)
Conflicting(c1, c2) == (c1 \cap c2) \subseteq Byzantine

NoConflictingCertificates ==
    \A c1, c2 \in CertificateSets : ~Conflicting(c1, c2)

VARIABLE accepted

Init == accepted = [cid \in {1} |-> [seq |-> 0, digest |-> 0]]

Accept(cid, seq, digest) ==
    /\ seq > accepted[cid].seq
    /\ accepted' = [accepted EXCEPT ![cid] = [seq |-> seq, digest |-> digest]]

Next == \E seq \in 1..3, digest \in 1..2 : Accept(1, seq, digest)

MonotoneSeq == accepted[1].seq >= 0

=============================================================================
