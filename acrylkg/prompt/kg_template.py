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

def get_fewshot_prompt_detailed():
    input_query = ""
    context = "Please generate a knowledge graph using only the given documents. \n"
    context += "Document : Steven Paul Jobs (February 24, 1955 – October 5, 2011) was an American business magnate, inventor, and investor best known as the co-founder of Apple. Jobs was also chairman and majority shareholder of Pixar, and the founder of NeXT. He was a pioneer of the personal computer revolution of the 1970s and 1980s, along with his early business partner and fellow Apple co-founder Steve Wozniak.\n"
    context += "Knowledge Graph : (Steve Jobs, was born on, February 24 1955), (Steve Jobs, died on, October 5 2011), (Steve Jobs, nationality, American), (Steve Jobs, was a, business magnate), (Steve Jobs, was a, inventor), (Steve Jobs, was a, investor), (Steve Jobs, co-founded, Apple), (Apple, is a, technology company), (Steve Jobs, was chairman of, Pixar), (Steve Jobs, was majority shareholder of, Pixar), (Pixar, is a, film studio), (Steve Jobs, founded, NeXT), (NeXT, is a, technology company), (Steve Jobs, was a pioneer in, personal computer revolution), (Personal computer revolution, occurred in, 1970s and 1980s), (Steve Jobs, business partner with, Steve Wozniak). (Steve Wozniak, co-founded, Apple)\n"
    context += "\n"
    context += "Please generate a knowledge graph using only the given documents. \n"
    context += "Document : Yi is regarded as one of the greatest naval commanders in history, with commentators praising his strategic vision, intelligence, innovations, and personality. Yi is celebrated as a national hero in Korea, with multiple landmarks, awards and towns named after him, as well as numerous films and documentaries centered on his exploits. His personal diaries, Nanjung Ilgi, covering a seven year period, are listed as part of UNESCO's Memory of the World.\n"
    context += "Knowledge Graph : (Yi Sun-sin, is regarded as, one of the greatest naval commanders in history), (Yi Sun-sin, praised for, strategic vision), (Yi Sun-sin, praised for, intelligence), (Yi Sun-sin, praised for, innovations), (Yi Sun-sin, praised for, personality), (Yi Sun-sin, celebrated as, national hero in Korea), (Multiple landmarks, named after, Yi Sun-sin), (Multiple awards, named after, Yi Sun-sin), (Towns, named after, Yi Sun-sin), (Numerous films, centered on, Yi Sun-sin's exploits), (Numerous documentaries, centered on, Yi Sun-sin's exploits), (Nanjung Ilgi, type of, personal diaries of Yi Sun-sin), (Nanjung Ilgi, covers, seven-year period), (Nanjung Ilgi, listed in, UNESCO's Memory of the World)\n"
    context += "\n"
    context += "Please generate a knowledge graph using only the given documents. \n"
    return input_query, context

def get_fewshot_prompt_detailed_ko():
    input_query = ""
    context = "지식 그래프를 천천히 자세히 생성해 주세요.\n"
    context += "문서 :첫 번째 문서: 스티븐 폴 잡스 (1955년 2월 24일 - 2011년 10월 5일)는 미국의 사업가, 발명가, 투자자로서, 애플의 공동 창립자로 가장 잘 알려져 있습니다. 잡스는 또한 픽사의 회장이자 다수 주주였으며, NeXT의 창립자이기도 했습니다. 그는 1970년대와 1980년대 개인 컴퓨터 혁명의 선구자였으며, 초기 사업 파트너이자 애플의 공동 창립자인 스티브 워즈니악과 함께 활동했습니다."
    context += "지식 그래프 : (스티브 잡스, 출생일, 1955년 2월 24일), (스티브 잡스, 사망일, 2011년 10월 5일), (스티브 잡스, 국적, 미국), (스티브 잡스, 직업, 사업가), (스티브 잡스, 직업, 발명가), (스티브 잡스, 직업, 투자자), (스티브 잡스, 공동 창업, 애플), (애플, 업종, 기술 회사), (스티브 잡스, 회장, 픽사), (스티브 잡스, 다수 주주, 픽사), (픽사, 업종, 영화 스튜디오), (스티브 잡스, 창립, NeXT), (NeXT, 업종, 기술 회사), (스티브 잡스, 선구자, 개인 컴퓨터 혁명), (개인 컴퓨터 혁명, 발생 시기, 1970년대 및 1980년대), (스티브 잡스, 사업 파트너, 스티브 워즈니악), (스티브 워즈니악, 공동 창립, 애플)"
    context += "\n"
    context += "문서 : 이순신은 역사상 가장 위대한 해군 지휘관 중 한 명으로 평가받으며, 그의 전략적 비전, 지능, 혁신, 인격을 높이 평가받습니다. 이순신은 한국의 국민 영웅으로 추앙받으며, 그의 이름을 딴 다수의 랜드마크, 상, 도시가 있으며 그의 업적을 중심으로 한 수많은 영화와 다큐멘터리가 제작되었습니다. 그의 개인 일기인 '난중일기'는 7년간의 기간을 다루며 유네스코 세계 기록 유산에 등재되어 있습니다."
    context += "지식 그래프 : (이순신, 평가, 역사상 가장 위대한 해군 지휘관 중 하나), (이순신, 높이 평가 받음, 전략적 비전), (이순신, 높이 평가 받음, 지능), (이순신, 높이 평가 받음, 혁신), (이순신, 높이 평가 받음, 인격), (이순신, 추앙 받음, 한국의 국민 영웅), (다수 랜드마크, 명명, 이순신), (다수 상, 명명, 이순신), (도시들, 명명, 이순신), (수많은 영화, 주제, 이순신의 업적), (수많은 다큐멘터리, 주제, 이순신의 업적), (난중일기, 유형, 이순신의 개인 일기), (난중일기, 기간, 7년), (난중일기, 등재, 유네스코 세계 기록 유산)"
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