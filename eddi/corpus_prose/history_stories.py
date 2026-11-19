"""History as narrative: what happened, why, and what followed.

History prose is valuable to a language model for a structural reason: it is
full of cause and consequence ("so", "because of that", "the result was"), and
it moves through time ("by then", "over the next century"). Those connectives
are what turn a list of facts into connected writing, and no amount of
chemistry prose teaches them.

Every claim here is a mainstream historical consensus stated plainly. Where
historians genuinely disagree, the sentence says so.
"""

from __future__ import annotations

from typing import List

SENTENCES: List[str] = [
    # ------------------------------------------------- early civilisations
    "Farming changed everything: once people could store grain, they settled in one place.",
    "Settled life allowed towns, because a surplus of food freed some people from farming.",
    "The first cities appeared in Mesopotamia between two rivers, where irrigation made the land productive.",
    "Writing began as accounting, because grain stores needed to be recorded.",
    "Cuneiform was pressed into clay with a reed, and the marks later grew into a full script.",
    "Egypt depended on the Nile flooding on schedule, and its calendar was built to predict it.",
    "The pyramids were engineering projects as much as monuments, built with ramps and enormous organisation.",
    "A pharaoh was both a king and a god, which is why obedience was not a political question.",
    "The Indus valley cities had drains and grid streets before most of Europe had either.",
    "Hammurabi's laws set out penalties in public, which limited what a ruler could do without justification.",
    "Iron replaced bronze slowly, because it is harder to work but far more common.",
    "Alphabetic writing made literacy cheaper, because a trader could learn a few dozen signs instead of thousands.",
    "Greece was a collection of competing cities rather than one country, which is why its politics were so inventive.",
    "Athenian democracy gave citizens a vote, but citizens excluded women, slaves and foreigners.",
    "The Persian empire governed through provinces and local rulers, which proved more durable than conquest alone.",
    "Alexander conquered quickly and died young, so his empire broke into pieces almost at once.",
    "Rome began as a city and grew by absorbing the peoples it defeated.",
    "The Roman Republic balanced two consuls, a senate and popular assemblies to stop any one person taking power.",
    "Roman roads, law and citizenship held an enormous empire together.",
    "Julius Caesar crossed the Rubicon with an army, which meant civil war in Roman terms.",
    "Caesar's assassination did not restore the republic; it produced another round of civil war.",
    "Augustus kept the forms of the republic while holding the substance of power.",
    "The Roman empire split into east and west because it was too large to govern from one place.",
    "The western empire fell to a mixture of invasion, inflation and political collapse.",
    "The eastern empire survived for another thousand years from Constantinople.",
    "Han China invented paper, which took a thousand years to reach Europe.",
    "Paper made writing cheap, and cheap writing made bureaucracy possible.",
    # --------------------------------------------------- medieval world
    "After the western empire fell, Europe lost the large scale trade and administration it had relied on.",
    "Monasteries preserved books by copying them, which is why so much ancient literature survived.",
    "The Islamic golden age translated and extended Greek science while Europe was recovering.",
    "Arabic numerals reached Europe through north Africa, and they made calculation far easier than Roman numerals.",
    "Algebra comes from an Arabic word, and its methods were developed in Baghdad.",
    "A guild controlled who could work a trade, which kept quality high and entry expensive.",
    "Feudalism exchanged land for military service, and it worked only while land was the main form of wealth.",
    "A medieval town had its own law, market and walls, which made it a world apart from the countryside.",
    "The Black Death killed perhaps a third of Europe, which made labour scarce and wages rise.",
    "When labour became scarce, serfdom weakened, because lords had to compete for workers.",
    "The printing press made books cheap enough for ordinary readers.",
    "Cheap books spread new ideas faster than any authority could suppress them.",
    "The Reformation began as an argument about indulgences and became a political division of Europe.",
    "Religious wars ended with a settlement that accepted that states could choose their own religion.",
    "Magna Carta limited the king by written law, and later generations read it as a foundation of liberty.",
    "A parliament began as a way for kings to raise taxes by consent.",
    # -------------------------------------------------- exploration and empires
    "Ocean navigation improved with the compass, better charts and ships that could sail against the wind.",
    "Columbus reached the Americas while looking for a western route to Asia.",
    "Contact between the continents moved crops, animals and diseases in both directions.",
    "Smallpox killed far more people in the Americas than warfare did.",
    "Silver from the Americas flowed to Europe and then to Asia, linking the world's economies.",
    "The transatlantic slave trade forcibly moved millions of people and shaped economies on both sides.",
    "The Columbian exchange brought potatoes and tomatoes to Europe and wheat to the Americas.",
    "Colonial empires extracted raw materials and sold finished goods back, which made the colonies dependent.",
    "Imperial competition produced a global system of trade and a global system of conflict.",
    "India's cotton industry was deliberately suppressed by colonial policy, which is why it industrialised late.",
    "The Ottoman, Mughal and Qing empires were large, rich and sophisticated at the same time as small Europe.",
    # ------------------------------------------------------ revolution and industry
    "The scientific revolution changed the method as much as the conclusions.",
    "An experiment that can be repeated by a stranger became the standard of proof.",
    "The Enlightenment argued that authority derived from reason and consent rather than from birth.",
    "Steam power freed industry from rivers, because an engine can be built anywhere fuel can be delivered.",
    "Factories concentrated workers and machines, which created both productivity and a new kind of poverty.",
    "Railways shrank distances and made national markets possible for the first time.",
    "Mass production lowered prices and standardised products.",
    "The American Revolution declared rights that were by then widely argued but rarely granted.",
    "The French Revolution destroyed an old order and then struggled to build a stable new one.",
    "Nationalism turned a shared language and history into a claim on a state.",
    "The industrial revolution spread unevenly, which created a gap between industrialised and other countries.",
    "Reform movements for workers, women and slaves all drew on the same language of natural rights.",
    "The abolition of slavery was achieved by argument, economics and violence together.",
    "Universal education followed industrialisation, because factories needed literate workers.",
    # --------------------------------------------------- the modern world
    "The first world war began with a diplomatic crisis and escalated through alliance commitments.",
    "Trench warfare made defence far stronger than attack, which is why the war lasted so long.",
    "The treaty that ended the war imposed reparations and territorial losses on Germany.",
    "Economic depression and the terms of peace made extreme politics attractive.",
    "The second world war killed around sixty million people, most of them civilians.",
    "The Holocaust was the systematic murder of six million Jews and millions of others.",
    "Nuclear weapons ended the war and began a new kind of peace built on deterrence.",
    "The cold war was fought through proxies, technology and ideology rather than direct battle.",
    "Decolonisation after 1945 created dozens of new states, often with borders drawn for imperial convenience.",
    "The Universal Declaration of Human Rights set out rights that no state had fully implemented.",
    "The European Union began as an attempt to make war between its members impractical.",
    "The internet began as a research network and became a general communication system within thirty years.",
    "The fall of the Berlin Wall showed that a system can look permanent and then dissolve in weeks.",
    "Globalisation joined markets while leaving politics national, which is a permanent source of friction.",
    "The green revolution raised crop yields and fed billions, at the cost of water and soil.",
    # ------------------------------------------------------ how history works
    "History is written from evidence, and the evidence is always incomplete.",
    "The winners write more of the record, but they do not write all of it.",
    "An ordinary record of a tax payment can tell us more than a royal monument.",
    "Contemporaries never know which events will matter, so they keep the wrong records.",
    "Hindsight makes decisions look obvious that were made in real uncertainty.",
    "Most historical change is slow, and the moments we remember as turning points were often late recognitions.",
    "Technology changes what is possible, and politics decides what is done with it.",
    "Disease, climate and harvest shaped history at least as much as kings did.",
    "A statistic from centuries ago is a reconstruction, not a measurement.",
    "Comparing different periods is hard because the questions people asked were different.",
]
