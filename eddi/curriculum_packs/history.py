"""History: dated events, people, inventions, and the shape of the last 5000 years.

Dates are the highest-risk facts in any curriculum: a model that sounds
confident will happily place the printing press in the wrong century. Each
event here is stored with a year as a plain number so it can be checked.

Years before the common era are written ``BCE`` so the sign is never
ambiguous. Anything whose date scholars dispute is stated as a range or left
to ``PROSE`` with the dispute named.
"""

from __future__ import annotations

from typing import List, Tuple

Fact = Tuple[str, str, str]

FACTS: List[Fact] = [
    # ------------------------------------------------- ancient civilisations
    ("agriculture", "began_years_ago", "10000"),
    ("writing", "invented_years_ago", "5200"),
    ("Mesopotamia", "located_in", "modern_Iraq"),
    ("Sumer", "invented", "cuneiform_writing"),
    ("cuneiform", "invented_by", "Sumer"),
    ("Egypt", "civilisation_began_year_BCE", "3100"),
    ("pyramid_of_Giza", "built_year_BCE", "2560"),
    ("pyramid_of_Giza", "built_for", "pharaoh_Khufu"),
    ("Great_Wall_of_China", "built_over_years", "2000"),
    ("Indus_Valley_Civilisation", "flourished_year_BCE", "2500"),
    ("Hammurabi", "created", "early_law_code"),
    ("Hammurabi", "ruled", "Babylon"),
    ("Tutankhamun", "ruled_year_BCE", "1332"),
    ("Ramses_II", "ruled_year_BCE", "1279"),
    ("Homer", "wrote", "the_Iliad_epic_poem"),
    ("the_Odyssey", "written_by", "Homer"),
    ("Olympic_Games", "first_held_year_BCE", "776"),
    ("Rome", "founded_year_BCE", "753"),
    ("Roman_Republic", "began_year_BCE", "509"),
    ("Julius_Caesar", "assassinated_year_BCE", "44"),
    ("Augustus", "first_emperor_of", "Rome"),
    ("Roman_Empire", "fell_year", "476"),
    ("Socrates", "taught", "Plato"),
    ("Plato", "taught", "Aristotle"),
    ("Aristotle", "taught", "Alexander_the_Great"),
    ("Alexander_the_Great", "died_year_BCE", "323"),
    ("Euclid", "wrote", "the_Elements"),
    ("Archimedes", "lived_year_BCE", "287"),
    ("Pythagoras", "lived_year_BCE", "570"),
    ("Confucius", "lived_year_BCE", "551"),
    ("Buddha", "lived_year_BCE", "563"),
    ("Jesus", "born_year_BCE", "4"),
    ("Hinduism", "originated", "India"),
    ("Buddhism", "founded_by", "Siddhartha_Gautama"),
    ("Islam", "founded_year", "610"),
    ("Islam", "founded_by", "Muhammad"),
    # ------------------------------------------------ medieval and early modern
    ("Byzantine_Empire", "fell_year", "1453"),
    ("Constantinople", "captured_by", "Ottoman_Empire"),
    ("Charlemagne", "crowned_year", "800"),
    ("printing_press", "invented_year", "1440"),
    ("printing_press", "invented_by", "Gutenberg"),
    ("Magna_Carta", "signed_year", "1215"),
    ("Black_Death", "peak_year", "1347"),
    ("Hundred_Years_War", "began_year", "1337"),
    ("Hundred_Years_War", "ended_year", "1453"),
    ("Columbus", "reached_Americas_year", "1492"),
    ("Magellan_expedition", "first_circumnavigation_year", "1522"),
    ("Reformation", "began_year", "1517"),
    ("Reformation", "started_by", "Martin_Luther"),
    ("Copernicus", "published_heliocentric_model_year", "1543"),
    ("Galileo", "improved", "the_telescope"),
    ("Kepler", "published_laws_year", "1609"),
    ("Newton", "published_principia_year", "1687"),
    ("Telescope", "invented_year", "1608"),
    ("Microscope", "invented_year", "1590"),
    ("Thirty_Years_War", "ended_year", "1648"),
    ("English_Civil_War", "began_year", "1642"),
    ("Glorious_Revolution", "year", "1688"),
    ("steam_engine", "improved_by", "James_Watt"),
    ("steam_engine", "improved_year", "1769"),
    ("Industrial_Revolution", "began_year", "1760"),
    ("Industrial_Revolution", "began_in", "Britain"),
    ("American_Declaration_of_Independence", "year", "1776"),
    ("French_Revolution", "began_year", "1789"),
    ("Napoleon", "became_emperor_year", "1804"),
    ("Napoleon", "defeated_year", "1815"),
    ("Battle_of_Waterloo", "year", "1815"),
]

PROSE: List[str] = [
    "History is the study of the recorded past, and it is written from evidence that is always incomplete.",
    "Agriculture began about ten thousand years ago, and it allowed people to settle in one place.",
    "Writing was invented about five thousand two hundred years ago in Mesopotamia to record grain and trade.",
    "Sumer invented cuneiform, one of the earliest writing systems, on clay tablets.",
    "The Great Pyramid of Giza was built around 2560 BCE as a tomb for the pharaoh Khufu.",
    "The Great Wall of China was built over roughly two thousand years by several dynasties.",
    "Hammurabi of Babylon created one of the earliest written law codes.",
    "The Olympic Games were first held in 776 BCE at Olympia in Greece.",
    "Rome was traditionally founded in 753 BCE, and the Roman Republic began in 509 BCE.",
    "Julius Caesar was assassinated in 44 BCE, which led to the rise of the Roman Empire under Augustus.",
    "The Western Roman Empire is usually taken to have fallen in 476 CE.",
    "Socrates taught Plato, Plato taught Aristotle, and Aristotle taught Alexander the Great.",
    "The Elements of Euclid organised geometry so well that it was a textbook for more than two thousand years.",
    "Archimedes lived in the third century BCE and worked on levers, buoyancy and the value of pi.",
    "Confucius and the Buddha were near contemporaries in the sixth century BCE.",
    "Jesus of Nazareth was probably born around 4 BCE, which is why the calendar is slightly off.",
    "Islam was founded in the seventh century CE, and its starting point is the migration to Medina in 622.",
    "The Byzantine Empire ended in 1453 when the Ottomans captured Constantinople.",
    "Charlemagne was crowned emperor in 800 CE, reviving the imperial title in western Europe.",
    "Gutenberg's printing press, around 1440, made books cheap and spread ideas faster than ever before.",
    "The Magna Carta, signed in 1215, limited the power of the English king for the first time in writing.",
    "The Black Death reached Europe in 1347 and killed perhaps a third of its population.",
    "Columbus reached the Americas in 1492 and opened permanent contact between the continents.",
    "The Magellan expedition completed the first circumnavigation of the globe in 1522.",
    "The Reformation began in 1517 when Martin Luther challenged the Catholic church.",
    "Copernicus published his heliocentric model in 1543, putting the Sun at the centre.",
    "Kepler published his first two laws of planetary motion in 1609.",
    "Newton published the Principia in 1687, stating his laws of motion and universal gravitation.",
    "The telescope was invented around 1608 and Galileo improved it almost immediately.",
    "The Thirty Years War ended in 1648 with the Peace of Westphalia.",
    "James Watt improved the steam engine in 1769, and the Industrial Revolution began in Britain around 1760.",
    "The United States declared independence in 1776, and the French Revolution began in 1789.",
    "Napoleon became emperor in 1804 and was finally defeated at Waterloo in 1815.",
]
