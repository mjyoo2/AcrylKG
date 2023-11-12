kg_template_1="""
"""

def get_base_prompt():
    input_query = "What Subject-Predicate-Object knowledge graphs are included in the following sentence? Please return the possible answers. Require the answer only in the form : [subject, predicate, object]\n"
    context = "Document : John works at Google. Google is a tech company. John, who is an engineer, designed a bridge that spans the wide river, and he worked with a team.\nKnowledge Graph : (John, works-at, Google), (Google, is, tech-company), (John, is, engineer), (John, designed, bridge), (bridge, spans, river), (John, worked-with, team)\n"
    return input_query, context

def get_prompt_1():
    input_query = ""
    context = "Document : Steve Jobs co-founded Apple, which is headquartered in Cupertino. Apple created the iPhone, which was launched in 2007.\nKnowledge Graph : (Steve Jobs, co-founded, Apple), (Apple, is headquartered in, Cupertino), (Apple, created, iPhone), (iPhone, was launched in, 2007)\n"
    return input_query, context

def get_prompt_2():
    input_query = ""
    context = "Document : Bionico is a dessert found in the Jalisco region of Mexico. The name of the leader in Jalisco is Jesus Casillas Romero. Another dessert is a cake.\nKnowledge Graph : (Mexico, country, Bionico), (Jalisco, region, Bionico), (Jesús_Casillas_Romero, leaderName, Jalisco), (Cake, dishVariation, Dessert), (Dessert, course, Bionico)\n"
    return input_query, context

def get_prompt_3():
    input_query = ""
    context = "Document : \nKnowledge Graph : (, , ), (, , ), (, , ), (, , ), (, , ), (, , )\n"
    return input_query, context

def get_fewshot_prompt():
    input_query = ""
    context = "Knowledge graph generation examples.\n"
    context += "Document : John works at Google. Google is a tech company. John, who is an engineer, designed a bridge that spans the wide river, and he worked with a team.\nKnowledge Graph : (John, works-at, Google), (Google, is, tech-company), (John, is, engineer), (John, designed, bridge), (bridge, spans, river), (John, worked-with, team)\n"
    context += "\n"
    context += "Document : Steve Jobs co-founded Apple, which is headquartered in Cupertino. Apple created the iPhone, which was launched in 2007.\nKnowledge Graph : (Steve Jobs, co-founded, Apple), (Apple, is headquartered in, Cupertino), (Apple, created, iPhone), (iPhone, was launched in, 2007)\n"
    context += "\n"
    context += "Document : Bionico is a dessert found in the Jalisco region of Mexico. The name of the leader in Jalisco is Jesus Casillas Romero. Another dessert is a cake.\nKnowledge Graph : (Mexico, country, Bionico), (Jalisco, region, Bionico), (Jesús_Casillas_Romero, leaderName, Jalisco), (Cake, dishVariation, Dessert), (Dessert, course, Bionico)\n"
    return input_query, context

def get_fewsot_prompt_QA():
    context = "Knowledge graph generation examples.\n"
    context += "Context : (John, works-at, Google)\n\n(Google, is, tech-company)\n\n(John, is, engineer)\n\n(John, designed, bridge)\n\n(bridge, spans, river)\n\n(John, worked-with, team)\n"
    context += "Question : Where did John work?\n"
    context += "Answer : Based on (John, work-at, Google), (Google, is, tech-company), Join work at Google, which is tech-company.\n"
    context += "\n"
    context += "Context : (Steve Jobs, co-founded, Apple)\n\n(Apple, is headquartered in, Cupertino)\n\n(Apple, created, iPhone)\n\n(iPhone, was launched in, 2007)\n"
    context += "Question : When was the iPhone released?\n"
    context += "Answer : Based on (iPhone, was launched in, 2007), iPhone is released in 2007\n"
    context += "\n"
    return context