"""
TextMind: Personality Data Module
Contains MBTI personality descriptions and trait definitions.
"""

# Dictionary of all 16 MBTI personality types with detailed information
personality_descriptions = {
    "INTJ": {
        "title": "The Architect",
        "description": "Strategic and independent thinkers who excel at long-term planning and implementing ideas. They are driven by their inner vision and have high standards for themselves and others.",
        "traits": ["Strategic", "Independent", "Logical", "Determined"],
        "celebrities": ["Elon Musk", "Friedrich Nietzsche", "Sherlock Holmes (fictional)"]
    },
    "INTP": {
        "title": "The Logician",
        "description": "Innovative problem-solvers who are fascinated by ideas and theoretical concepts. They enjoy analyzing complex systems and pursuing knowledge for its own sake.",
        "traits": ["Analytical", "Creative", "Curious", "Objective"],
        "celebrities": ["Albert Einstein", "Isaac Newton", "Bill Gates"]
    },
    "ENTJ": {
        "title": "The Commander",
        "description": "Natural-born leaders who are decisive, strategic, and driven to achieve ambitious goals. They excel at organizing people and resources to accomplish their vision.",
        "traits": ["Leadership", "Strategic", "Ambitious", "Efficient"],
        "celebrities": ["Donald Trump", "Margaret Thatcher", "Steve Jobs"]
    },
    "ENTP": {
        "title": "The Debater",
        "description": "Innovative and enthusiastic explorers of ideas who enjoy intellectual debate and challenging the status quo. They are quick-thinking and adaptable problem-solvers.",
        "traits": ["Innovative", "Curious", "Argumentative", "Energetic"],
        "celebrities": ["Mark Zuckerberg", "Oscar Wilde", "Chandler Bing (fictional)"]
    },
    "INFJ": {
        "title": "The Advocate",
        "description": "Insightful and compassionate individuals with a deep understanding of human motivation. They are driven by their values and desire to make a positive impact on the world.",
        "traits": ["Insightful", "Compassionate", "Principled", "Determined"],
        "celebrities": ["Nelson Mandela", "Oprah Winfrey", "Atticus Finch (fictional)"]
    },
    "INFP": {
        "title": "The Mediator",
        "description": "Idealistic and creative individuals who are guided by their values and personal authenticity. They are deeply empathetic and seek meaningful connections with others.",
        "traits": ["Creative", "Idealistic", "Empathetic", "Flexible"],
        "celebrities": ["J.R.R. Tolkien", "William Shakespeare", "Luna Lovegood (fictional)"]
    },
    "ENFJ": {
        "title": "The Protagonist",
        "description": "Charismatic and inspiring leaders who are passionate about helping others reach their potential. They are natural communicators with strong interpersonal skills.",
        "traits": ["Charismatic", "Empathetic", "Inspiring", "Organized"],
        "celebrities": ["Barack Obama", "Oprah Winfrey", "Ben Wyatt (fictional)"]
    },
    "ENFP": {
        "title": "The Campaigner",
        "description": "Enthusiastic and outgoing individuals who love exploring new ideas and meeting new people. They are spontaneous, creative, and driven by their passions.",
        "traits": ["Enthusiastic", "Creative", "Spontaneous", "Sociable"],
        "celebrities": ["Ellen DeGeneres", "Robin Williams", "Phoebe Buffay (fictional)"]
    },
    "ISTJ": {
        "title": "The Logistician",
        "description": "Practical and fact-oriented individuals who are reliable and responsible. They excel at organizing systems and following through on commitments with precision.",
        "traits": ["Responsible", "Practical", "Dependable", "Organized"],
        "celebrities": ["George Washington", "Queen Elizabeth II", "Severus Snape (fictional)"]
    },
    "ISFJ": {
        "title": "The Defender",
        "description": "Warm and caring individuals who are dedicated to serving others and maintaining harmony. They are practical helpers who remember details about people they care for.",
        "traits": ["Caring", "Loyal", "Practical", "Humble"],
        "celebrities": ["Queen Victoria", "Selena Gomez", "Molly Weasley (fictional)"]
    },
    "ESTJ": {
        "title": "The Executive",
        "description": "Practical leaders who are focused on efficiency and order. They are decisive, disciplined, and excel at managing people and processes to achieve clear objectives.",
        "traits": ["Practical", "Leadership", "Dutiful", "Efficient"],
        "celebrities": ["Franklin D. Roosevelt", "Judge Judy", "Miranda Priestly (fictional)"]
    },
    "ESFJ": {
        "title": "The Consul",
        "description": "Sociable and caring individuals who are focused on supporting others and maintaining group harmony. They are loyal, responsible, and excel at organizing social events.",
        "traits": ["Loyal", "Caring", "Social", "Responsible"],
        "celebrities": ["Taylor Swift", "Bill Clinton", "Laverne DeFazio (fictional)"]
    },
    "ISTP": {
        "title": "The Virtuoso",
        "description": "Practical and observant individuals who are skilled at understanding how things work. They are independent, logical, and adaptable problem-solvers.",
        "traits": ["Practical", "Logical", "Independent", "Adaptable"],
        "celebrities": ["Tom Cruise", "Tiger Woods", "Han Solo (fictional)"]
    },
    "ISFP": {
        "title": "The Adventurer",
        "description": "Artistic and sensitive individuals who are driven by their personal values and aesthetic appreciation. They are spontaneous, loyal friends with a love for exploration.",
        "traits": ["Artistic", "Sensitive", "Spontaneous", "Loyal"],
        "celebrities": ["Britney Spears", "Cher", "Ariel (fictional)"]
    },
    "ESTP": {
        "title": "The Entrepreneur",
        "description": "Energetic and pragmatic individuals who are focused on immediate results and enjoying life. They are natural risk-takers who excel in dynamic, high-energy environments.",
        "traits": ["Energetic", "Pragmatic", "Risk-taking", "Adaptable"],
        "celebrities": ["Donald Trump", "Madonna", "Jack Sparrow (fictional)"]
    },
    "ESFP": {
        "title": "The Entertainer",
        "description": "Outgoing and spontaneous individuals who love being the center of attention and creating excitement. They are natural performers with excellent people skills.",
        "traits": ["Outgoing", "Spontaneous", "Charismatic", "Fun-loving"],
        "celebrities": ["Marilyn Monroe", "Jamie Foxx", "Phoebe Buffay (fictional)"]
    }
}

# Dictionary explaining the four MBTI dimensions and their components
trait_definitions = {
    "Mind": {
        "title": "Mind: How You Direct Your Energy",
        "I": {
            "letter": "I",
            "word": "Introverted",
            "definition": "You direct your energy inward. You recharge through alone time, deep thinking, and small group interactions. You prefer observing before acting."
        },
        "E": {
            "letter": "E",
            "word": "Extraverted",
            "definition": "You direct your energy outward. You recharge through social interaction, external stimulation, and action. You prefer engaging with the world directly."
        }
    },
    "Energy": {
        "title": "Energy: How You Perceive Information",
        "S": {
            "letter": "S",
            "word": "Sensing",
            "definition": "You focus on the present, concrete facts, and practical details. You trust your five senses and prefer realistic, hands-on approaches to problem-solving."
        },
        "N": {
            "letter": "N",
            "word": "Intuition",
            "definition": "You focus on patterns, possibilities, and the bigger picture. You trust your instincts and enjoy abstract thinking, imagination, and exploring 'what could be'."
        }
    },
    "Nature": {
        "title": "Nature: How You Make Decisions",
        "T": {
            "letter": "T",
            "word": "Thinking",
            "definition": "You make decisions based on objective logic and impersonal analysis. You prioritize truth and fairness, and tend to be straightforward in your approach."
        },
        "F": {
            "letter": "F",
            "word": "Feeling",
            "definition": "You make decisions based on personal values and how they affect people. You prioritize harmony and empathy, considering the human impact of choices."
        }
    },
    "Tactics": {
        "title": "Tactics: How You Organize Your Life",
        "J": {
            "letter": "J",
            "word": "Judging",
            "definition": "You prefer structure, planning, and closure. You like making decisions, setting schedules, and organizing your environment to reduce uncertainty."
        },
        "P": {
            "letter": "P",
            "word": "Perceiving",
            "definition": "You prefer flexibility, spontaneity, and keeping options open. You adapt easily to change, enjoy exploring possibilities, and prefer going with the flow."
        }
    }
}
