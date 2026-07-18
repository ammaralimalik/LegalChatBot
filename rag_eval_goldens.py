"""Hand-crafted Pakistani-law goldens for RAG evaluation."""

from deepeval.dataset import Golden


GOLDENS = [
    Golden(
        input="What conditions make an agreement a valid contract under Pakistani law?",
        expected_output=(
            "Under section 10 of the Contract Act, 1872, an agreement is a contract "
            "when it is made by competent parties with free consent, for lawful "
            "consideration and a lawful object, and is not expressly declared void."
        ),
    ),
    Golden(
        input="Who is competent to contract under the Contract Act, 1872?",
        expected_output=(
            "Under section 11 of the Contract Act, a person is competent to contract "
            "if they are of the age of majority according to the law to which they "
            "are subject, are of sound mind, and are not disqualified from "
            "contracting by any law to which they are subject."
        ),
    ),
    Golden(
        input="What is free consent under the Contract Act, 1872?",
        expected_output=(
            "Section 14 provides that consent is free when it is not caused by "
            "coercion, undue influence, fraud, misrepresentation, or mistake, "
            "subject to the provisions of the Act."
        ),
    ),
    Golden(
        input="What happens to an agreement made without consideration in Pakistan?",
        expected_output=(
            "Section 25 generally makes an agreement without consideration void, "
            "subject to statutory exceptions such as certain written and registered "
            "agreements made out of natural love and affection, promises to "
            "compensate for voluntary acts, and written promises to pay "
            "time-barred debts."
        ),
    ),
    Golden(
        input="When is the consideration or object of an agreement unlawful?",
        expected_output=(
            "Under section 23 of the Contract Act, consideration or an object is "
            "unlawful if it is forbidden by law, defeats the provisions of law, is "
            "fraudulent, involves injury to a person or property, or is regarded by "
            "the court as immoral or opposed to public policy."
        ),
    ),
    Golden(
        input="What is the effect of an agreement to do an impossible act?",
        expected_output=(
            "Section 56 of the Contract Act states that an agreement to do an act "
            "impossible in itself is void. A contract also becomes void when a "
            "supervening event makes performance impossible or unlawful, unless the "
            "event was caused by the promisor."
        ),
    ),
    Golden(
        input="What compensation is available for breach of contract in Pakistan?",
        expected_output=(
            "Section 73 of the Contract Act, 1872 permits compensation for loss or "
            "damage that naturally arose in the usual course from the breach, or "
            "which the parties knew when contracting was likely to result. Remote "
            "and indirect loss is not compensable."
        ),
    ),
    Golden(
        input="What jurisdiction do civil courts have under the Code of Civil Procedure?",
        expected_output=(
            "Section 9 of the Code of Civil Procedure provides that civil courts "
            "have jurisdiction to try all suits of a civil nature except suits of "
            "which their cognizance is expressly or impliedly barred."
        ),
    ),
    Golden(
        input="When can a civil suit be stayed because of an earlier suit?",
        expected_output=(
            "Section 10 of the Code of Civil Procedure requires a court not to "
            "proceed with trial of a suit when the matter in issue is directly and "
            "substantially in issue in a previously instituted suit between the "
            "same parties, or parties claiming under them, before a competent court."
        ),
    ),
    Golden(
        input="What is the rule of res judicata under the Code of Civil Procedure?",
        expected_output=(
            "Section 11 of the Code of Civil Procedure bars a court from trying an "
            "issue that has already been directly and substantially decided between "
            "the same parties, or parties claiming under them, by a competent court "
            "in a former suit."
        ),
    ),
    Golden(
        input="What does the Pakistani Constitution say about equality before law?",
        expected_output=(
            "Article 25 of the Constitution provides that all citizens are equal "
            "before law and entitled to equal protection of law. It also states "
            "that there shall be no discrimination on the basis of sex."
        ),
    ),
    Golden(
        input="What protection does Article 9 provide in Pakistan?",
        expected_output=(
            "Article 9 provides that no person may be deprived of life or liberty "
            "save in accordance with law."
        ),
    ),
    Golden(
        input="What safeguards apply when a person is arrested in Pakistan?",
        expected_output=(
            "Article 10 requires an arrested person to be informed of the grounds "
            "of arrest as soon as possible and permits consultation with and "
            "defence by a legal practitioner of their choice. The person must be "
            "produced before a magistrate within twenty-four hours of arrest, "
            "excluding necessary travel time, unless further detention is "
            "authorized by a magistrate."
        ),
    ),
    Golden(
        input="What does the Constitution protect under Article 14?",
        expected_output=(
            "Article 14 declares the dignity of man and, subject to law, the "
            "privacy of home inviolable. It also prohibits torture for the purpose "
            "of extracting evidence."
        ),
    ),
    Golden(
        input="What constitutional protection exists for freedom of speech in Pakistan?",
        expected_output=(
            "Article 19 guarantees every citizen the right to freedom of speech and "
            "expression and freedom of the press, subject to reasonable restrictions "
            "imposed by law in the interests listed in the Constitution."
        ),
    ),
    Golden(
        input="What right does Article 18 of the Constitution provide?",
        expected_output=(
            "Article 18 gives every citizen the right to enter a lawful profession "
            "or occupation and to conduct any lawful trade or business, subject to "
            "qualifications and reasonable restrictions imposed by law."
        ),
    ),
    Golden(
        input="What is the effect of common intention under the Pakistan Penal Code?",
        expected_output=(
            "Section 34 provides that when a criminal act is done by several "
            "persons in furtherance of the common intention of all, each person is "
            "liable for that act as if they had done it alone."
        ),
    ),
    Golden(
        input="When is an act done in private defence not an offence under the Pakistan Penal Code?",
        expected_output=(
            "Section 96 of the Pakistan Penal Code provides that nothing is an "
            "offence which is done in the exercise of the right of private defence. "
            "That right remains subject to the statutory limits on its exercise."
        ),
    ),
    Golden(
        input="What is abetment under the Pakistan Penal Code?",
        expected_output=(
            "Section 107 defines abetment as instigating a person to do a thing, "
            "engaging in a conspiracy for the doing of that thing with an act or "
            "illegal omission in pursuance of it, or intentionally aiding the "
            "doing of that thing."
        ),
    ),
    Golden(
        input="How does unsoundness of mind affect criminal liability under the Pakistan Penal Code?",
        expected_output=(
            "Section 84 provides that an act is not an offence when, at the time of "
            "doing it, a person is incapable because of unsoundness of mind of "
            "knowing the nature of the act or that it is wrong or contrary to law."
        ),
    ),
    Golden(
        input="When is communication of a proposal or acceptance complete under the Contract Act?",
        expected_output=(
            "Section 4 provides that communication of a proposal is complete when it "
            "comes to the knowledge of the person to whom it is made. Communication "
            "of acceptance is complete as against the proposer when it is put in a "
            "course of transmission to him so as to be out of the acceptor's power, "
            "and as against the acceptor when it comes to the knowledge of the proposer."
        ),
    ),
    Golden(
        input="What must acceptance of a proposal be under the Contract Act, 1872?",
        expected_output=(
            "Section 7 requires acceptance to be absolute and unqualified, and "
            "expressed in some usual and reasonable manner unless the proposal "
            "prescribes a particular manner of acceptance."
        ),
    ),
    Golden(
        input="What is coercion under the Contract Act, 1872?",
        expected_output=(
            "Section 15 defines coercion as committing or threatening to commit any "
            "act forbidden by the Penal Code, or unlawfully detaining or threatening "
            "to detain any property, to the prejudice of any person, with the "
            "intention of causing that person to enter into an agreement."
        ),
    ),
    Golden(
        input="What is undue influence under the Contract Act, 1872?",
        expected_output=(
            "Section 16 provides that a contract is induced by undue influence where "
            "the relations between the parties are such that one party can dominate "
            "the will of the other and uses that position to obtain an unfair "
            "advantage. A person is deemed to dominate another's will where he holds "
            "real or apparent authority over the other, stands in a fiduciary "
            "relation, or contracts with a person whose mental capacity is affected "
            "by age, illness, or mental or bodily distress."
        ),
    ),
    Golden(
        input="What is fraud under the Contract Act, 1872?",
        expected_output=(
            "Section 17 defines fraud to include acts such as suggesting as fact what "
            "is not true by one who does not believe it to be true, actively "
            "concealing a fact, making a promise without intention to perform, any "
            "other act fitted to deceive, or any act or omission declared fraudulent "
            "by law, committed with intent to deceive or induce entry into a contract."
        ),
    ),
    Golden(
        input="What is sound mind for the purposes of contracting under Pakistani law?",
        expected_output=(
            "Section 12 of the Contract Act provides that a person is of sound mind "
            "for contracting if, at the time of making the contract, he is capable of "
            "understanding it and forming a rational judgment as to its effect on "
            "his interests."
        ),
    ),
    Golden(
        input="What is a contingent contract under the Contract Act, 1872?",
        expected_output=(
            "Section 31 defines a contingent contract as a contract to do or not to "
            "do something if some event, collateral to the contract, does or does not "
            "happen."
        ),
    ),
    Golden(
        input="What happens when a party refuses to perform a contract wholly?",
        expected_output=(
            "Section 39 of the Contract Act provides that when a party refuses to "
            "perform, or disables himself from performing, his promise in its "
            "entirety, the promisee may put an end to the contract unless he has "
            "signified by words or conduct his acquiescence in its continuance."
        ),
    ),
    Golden(
        input="How are penalty clauses treated in breach of contract under Pakistani law?",
        expected_output=(
            "Section 74 of the Contract Act provides that where a contract names a "
            "sum payable on breach or contains another stipulation by way of penalty, "
            "the party complaining of breach is entitled to reasonable compensation "
            "not exceeding the amount named or penalty stipulated, whether or not "
            "actual damage is proved."
        ),
    ),
    Golden(
        input="Where may a civil suit be instituted under the Code of Civil Procedure?",
        expected_output=(
            "Section 20 of the Code of Civil Procedure provides that, subject to "
            "limitations, every suit shall be in a court within the local limits of "
            "whose jurisdiction the defendant, or any of the defendants where there "
            "are more than one, actually and voluntarily resides, carries on business, "
            "or personally works for gain at the commencement of the suit."
        ),
    ),
    Golden(
        input="When can a court grant a temporary injunction in a civil suit?",
        expected_output=(
            "Order 39 of the Code of Civil Procedure allows a plaintiff in a suit "
            "for restraining breach of contract or other injury to apply for a "
            "temporary injunction, before or after judgment, to restrain the defendant "
            "from committing the breach or injury complained of. The court may grant "
            "the injunction on such terms as to duration, security, or otherwise as "
            "it thinks fit."
        ),
    ),
    Golden(
        input="When does a second appeal lie to the High Court under the Code of Civil Procedure?",
        expected_output=(
            "Section 100 of the Code of Civil Procedure provides that, save where "
            "otherwise provided, an appeal lies to the High Court from every decree "
            "passed in first appeal by a subordinate court on grounds including that "
            "the decision is contrary to law or usage having the force of law, fails "
            "to determine a material issue of law, or involves a substantial error "
            "or defect in procedure that may have affected the decision on the merits."
        ),
    ),
    Golden(
        input="What is the revisional jurisdiction of the High Court under the Code of Civil Procedure?",
        expected_output=(
            "Section 115 empowers the High Court to call for the record of any case "
            "decided by a subordinate court in which no appeal lies, and if that "
            "court appears to have exercised jurisdiction not vested in it, failed "
            "to exercise vested jurisdiction, or acted illegally or with material "
            "irregularity, the High Court may make such order as it thinks fit."
        ),
    ),
    Golden(
        input="When may an objection to the place of suing be raised in a civil case?",
        expected_output=(
            "Section 21 of the Code of Civil Procedure provides that no objection as "
            "to the place of suing shall be allowed by an appellate or revisional "
            "court unless the objection was taken in the court of first instance at "
            "the earliest possible opportunity and, where issues are settled, at or "
            "before that settlement, and unless there has been consequent failure of "
            "justice."
        ),
    ),
    Golden(
        input="What does Article 8 of the Pakistani Constitution say about fundamental rights?",
        expected_output=(
            "Article 8 provides that any law or custom having the force of law "
            "inconsistent with fundamental rights is void to the extent of the "
            "inconsistency, and the State shall not make any law that takes away or "
            "abridges those rights."
        ),
    ),
    Golden(
        input="What right does Article 10A of the Pakistani Constitution guarantee?",
        expected_output=(
            "Article 10A provides that for the determination of civil rights and "
            "obligations or in any criminal charge, a person is entitled to a fair "
            "trial and due process."
        ),
    ),
    Golden(
        input="What does the Pakistani Constitution say about slavery and forced labour?",
        expected_output=(
            "Article 11 declares that slavery is non-existent and forbidden, no law "
            "shall permit or facilitate its introduction, and all forms of forced "
            "labour and traffic in human beings are prohibited."
        ),
    ),
    Golden(
        input="What freedom of movement does Article 15 of the Pakistani Constitution provide?",
        expected_output=(
            "Article 15 gives every citizen the right to remain in, and subject to "
            "reasonable restrictions imposed by law in the public interest, enter and "
            "move freely throughout Pakistan and to reside and settle in any part "
            "thereof."
        ),
    ),
    Golden(
        input="What right does Article 16 of the Pakistani Constitution protect?",
        expected_output=(
            "Article 16 gives every citizen the right to assemble peacefully and "
            "without arms, subject to reasonable restrictions imposed by law in the "
            "interest of public order."
        ),
    ),
    Golden(
        input="What right to form associations does Article 17 of the Pakistani Constitution provide?",
        expected_output=(
            "Article 17 gives every citizen the right to form associations or unions, "
            "subject to reasonable restrictions imposed by law in the interest of the "
            "sovereignty or integrity of Pakistan, public order, or morality."
        ),
    ),
    Golden(
        input="What is the right to information under the Pakistani Constitution?",
        expected_output=(
            "Article 19A provides that every citizen has the right to have access to "
            "information in all matters of public importance, subject to regulation "
            "and reasonable restrictions imposed by law."
        ),
    ),
    Golden(
        input="What protection does Article 24 of the Pakistani Constitution give to property?",
        expected_output=(
            "Article 24 provides that no person shall be deprived of property save in "
            "accordance with law, and no property shall be compulsorily acquired or "
            "taken possession of except for a public purpose and by authority of law "
            "that provides for compensation."
        ),
    ),
    Golden(
        input="What is qatl-e-amd under the Pakistan Penal Code?",
        expected_output=(
            "Section 300 defines qatl-e-amd as causing the death of a person with "
            "intention to cause death, with intention to cause bodily injury by an "
            "act likely in the ordinary course of nature to cause death, or with "
            "knowledge that the act is so imminently dangerous that it must in all "
            "probability cause death."
        ),
    ),
    Golden(
        input="What is the punishment for qatl-e-amd under the Pakistan Penal Code?",
        expected_output=(
            "Section 302 provides that whoever commits qatl-e-amd shall, subject to "
            "the provisions of that Chapter, be punished with death as qisas or with "
            "death or imprisonment for life as ta'zir having regard to the facts and "
            "circumstances of the case."
        ),
    ),
    Golden(
        input="What is theft under the Pakistan Penal Code?",
        expected_output=(
            "Section 378 defines theft as dishonestly moving movable property out of "
            "the possession of any person without that person's consent, in order to "
            "take such property."
        ),
    ),
    Golden(
        input="What is the offence of cheating under section 420 of the Pakistan Penal Code?",
        expected_output=(
            "Section 420 punishes whoever cheats and thereby dishonestly induces the "
            "person deceived to deliver property, or to make, alter, or destroy a "
            "valuable security or anything capable of being converted into a valuable "
            "security, with imprisonment up to seven years and fine."
        ),
    ),
    Golden(
        input="What is the punishment for rape under the Pakistan Penal Code?",
        expected_output=(
            "Section 376 provides that whoever commits rape shall be punished with "
            "death or imprisonment of either description for a term of not less than "
            "ten years and not more than twenty-five years, and shall also be liable "
            "to fine."
        ),
    ),
    Golden(
        input="What is criminal conspiracy under the Pakistan Penal Code?",
        expected_output=(
            "Section 120A defines criminal conspiracy as an agreement by two or more "
            "persons to do or cause to be done an illegal act, or a legal act by "
            "illegal means. Section 120B provides punishment for parties to a criminal "
            "conspiracy to commit an offence."
        ),
    ),
    Golden(
        input="How is attempt to commit an offence punished under the Pakistan Penal Code?",
        expected_output=(
            "Section 511 provides that whoever attempts to commit an offence "
            "punishable with imprisonment, and does any act towards its commission, "
            "may where no express provision exists be punished with imprisonment up "
            "to one-half of the longest term provided for that offence, or with fine "
            "or both."
        ),
    ),
    Golden(
        input="What is dacoity under the Pakistan Penal Code?",
        expected_output=(
            "Section 391 provides that when five or more persons conjointly commit or "
            "attempt to commit robbery, or aid such commission or attempt so that "
            "the total number is five or more, every such person commits dacoity. "
            "Section 395 punishes dacoity with imprisonment for life or rigorous "
            "imprisonment for four to ten years and fine."
        ),
    ),
]
