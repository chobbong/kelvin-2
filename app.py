import streamlit as st
import matplotlib.pyplot as plt
from openpyxl import load_workbook
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import matplotlib.font_manager as fm

@st.cache_data
def fontRegistered():
    font_dirs = [os.getcwd() + '/font_directory']
    font_files = fm.findSystemFonts(fontpaths=font_dirs)

    for font_file in font_files:
        
        fm.fontManager.addfont(font_file)

    fm._load_fontmanager(try_read_cache=False)

# 파일 업로더 위젯
file_path = st.file_uploader("파일을 선택해주세요", type=['xlsx'])
xl_new = pd.read_excel(file_path)

# 'character' 시트와 'score' 시트 로드
character_sheet = pd.read_excel(file_path, sheet_name='character')
score_sheet = pd.read_excel(file_path, sheet_name='score')

character_sheet.columns = character_sheet.iloc[0]
character_sheet = character_sheet.drop(character_sheet.index[0])
type_selected = pd.read_excel(file_path, sheet_name='type')
type_selected = type_selected['유형'][0]

reverse_dict = character_sheet.set_index('Question')['Reverse'].to_dict()

# 'score' 시트에서 첫 번째 컬럼 (index) 제거
score_sheet = score_sheet.drop(columns=score_sheet.columns[0])

# 재산정된 점수를 저장할 DataFrame 생성
score_adjusted = score_sheet.copy()

reverse_dict_int = {key: int(value) for key, value in reverse_dict.items()}

# score_sheet의 원본 점수를 바탕으로 역점 반영 점수 재산정
def adjust_scores_properly(row):
    for question, reverse in reverse_dict_int.items():
        if question in row.index:  # score_sheet에 해당 문제가 있는 경우만 처리
            if reverse == 1:  # 역점이 '1'인 경우
                row[question] = 6 - row[question]  # 원 점수를 6에서 빼서 반영
    return row


# 'score_sheet'의 복사본에서 학생 이름을 제외한 모든 점수에 대해 재산정 함수 적용
score_sheet_adjusted_proper = score_sheet.drop(columns=['Name']).apply(adjust_scores_properly, axis=1)

# 학생 이름 컬럼 다시 추가
score_sheet_adjusted_proper['Name'] = score_sheet['Name']

# 점수 재조정 후 데이터를 학생 이름을 인덱스로 하여 전치
score_sheet_adjusted_proper_transposed = score_sheet_adjusted_proper.set_index('Name').transpose()

# 데이터프레임을 딕셔너리로 전환합니다.
df_dict = score_sheet_adjusted_proper_transposed.to_dict(orient='index')

# 'character' 시트의 색깔 정보를 사용하여 문제 ID를 색깔별로 그룹화합니다.
color_groups_new = character_sheet[['Question', 'Color']].set_index('Question')['Color'].to_dict()


# 학생별로 각 색깔에 대한 합산 점수를 계산하기 위한 딕셔너리를 초기화합니다.
student_color_scores_new = {student: {color: 0 for color in set(color_groups_new.values())} for student in score_sheet_adjusted_proper_transposed.columns}

# 점수 데이터를 사용하여 각 문제에 대한 점수를 가져와 색깔 그룹에 맞춰 합산합니다.
for problem_id, scores in df_dict.items():
    color = color_groups_new.get(problem_id)
    for student, score in scores.items():
        student_color_scores_new[student][color] += score
        
student_color_scores_df_uploaded = pd.DataFrame(student_color_scores_new).T

# 각 학생별로 점수가 높은 순으로 색깔을 정렬합니다.
sorted_scores_df_uploaded = student_color_scores_df_uploaded.apply(lambda x: x.sort_values(ascending=False), axis=1)

#--------------------------------------------표시------------------------------------------------
st.markdown("""
   ##### (유)Edupia 홍쌤 색채심리 연구소
""")
# Streamlit 앱의 제목 설정
st.markdown(f"<h2 style='text-align: left; color: black;'>휴먼컬러/정서 행동 특성 검사 결과({type_selected})</h2>", unsafe_allow_html=True)

# DataFrame에서 학생 이름 목록 추출
students_list = sorted_scores_df_uploaded.index.tolist()

# st.selectbox를 사용하여 학생 이름 선택
selected_student = st.selectbox("학생 이름을 선택하세요:", students_list)

# 선택된 학생의 점수를 DataFrame 형태로 표시
def get_sorted_scores(student_name):
    if student_name in sorted_scores_df_uploaded.index:
        student_scores = sorted_scores_df_uploaded.loc[student_name].sort_values(ascending=False)
        return pd.DataFrame(student_scores).T
    else:
        return pd.DataFrame({"Error": ["Student name not found."]})
   
st.markdown("""
### 1. 휴먼컬러 결과
#### 검사자별 성격 유형 <최고점 50~최하점 10>
""")

# 결과 표시
student_scores_df = get_sorted_scores(selected_student)
st.write(student_scores_df)

st.write("""
#### 최고 점수와 최하 점수 차이 분석 기준
- **7점 미만의 차이:** 소극적인 참여 또는 진실성 결여 가능성이 있음
- **7점~14점 차이:** 다양성, 수용성이 높으나 주관성, 가치관 정립이 낮을 수 있음
- **15점 이상 차이:** 또렷한 성격적 특성을 가지고 있음

#### 점수 배점별 해석 기준
- **최고 점수 컬러:** 검사자의 성격적 고유한 주요 특성
- **중간 점수 컬러:** 검사자의 성격적 중간 에너지 사용
- **최하 점수 컬러:** 검사자의 소극적 성격 특성
""")
def main():
    fontRegistered()
    fontname = 'Malgun Gothic'
    # 한글 폰트 설정
    plt.rc('font', family=fontname)
    # 그래프 그리기
    if not student_scores_df.empty and 'Error' not in student_scores_df.columns:
        # Extract scores and columns
        colors = student_scores_df.columns.values
        scores = student_scores_df.iloc[0].values
    
        # Create the linear plot
        plt.figure(figsize=(10, 6))
        plt.plot(colors, scores, marker='o', linestyle='-', color='orange')
    
        # Annotate each point with its score
        for i, score in enumerate(scores):
            plt.text(colors[i], score + 0.5, str(score), ha='center', va='bottom', fontsize=10)
    
        # Adding title and labels
        plt.title(f'Score Distribution for {selected_student}')
        plt.xlabel('Subject')
        plt.ylabel('Score')
        plt.grid(True)
        
         # 설정한 y축 범위와 눈금 간격을 적용
        plt.ylim(10, 50)  # y축의 최소값과 최대값 설정
        plt.yticks(range(10, 51, 5))  # y축에 표시될 눈금 설정
    
        # Display the plot in Streamlit
        st.pyplot(plt)
        
if __name__ == "__main__":
    main()
color_info_dict_full = {'보라': {'요약': '독창성과 신비로운 보라',
  '성격상 강점': '-자신을 특별하다고 여기며 독특한 매력을 인정받고 싶어함.\n-정신적이고 신비로운 것에 끌리며 탐구력을 보임.\n-믿음과 신뢰를 구축한 사람은 끝까지 함께하며 책임감을 가짐.\n-변화와 안정감 두 개의 마음을 가짐.\n-관계 중시.\n-자신이 선택한 것에 책임감을 가짐.',
  '성격상 약점': '-양가감정이 있어 변덕스럽거나 감정기복이 있음.\n-정신적인 것을 너무 추구하면 고독, 불안, 우울이 있을 수 있음.\n-정체성의 혼란이 있을 수 있음.\n-타인과 구별되는 우월감을 가짐.',
  '추천 직업': '행위예술가, 연극배우, 수필가, 극작가, 시인, 종교인, 상담사, 치료사, 사진가, 건축가, 실내디자이너, 일러스트레이터, 문학 연구자, 사회학자'},
 '분홍': {'요약': '상냥하며 매력적인 유형',
  '성격상 강점': '-온화하고 따뜻한 성격의 소유자.\n-타인의 감정에 대해 민감하며 공감능력이 뛰어남.\n-감수성이 풍부하며 애교가 많음.\n-타인이 나를 어떻게 생각하는지에 대한 관심이 많음.\n-보호하거나 보호받고 싶어함.\n-협조적이며 순응적인 성격.',
  '성격상 약점': '-즉흥적 감정으로 표현하는 경우가 있어 신중함이 적음.\n-사랑, 관심 받고 싶은 마음이 커서 사람과의 관계에서 자주 스트레스를 받음.\n-독립심이 적고 의존적인 성향임.\n-허영심이 있기도 하며 무책임함이 있을 수 있음.',
  '추천 직업': '예술가, 음악가, 공예가, 마술사, 댄서, 모델, 반려견 사업, 애완견 미용, 플로리스트, 호텔리어, 카페, 웨딩플래너, 메이크업아티스트, 간호사, 보육교사, \n어린이 관련 직업'},
 '초록': {'요약': '예의바르며 성실한 안정형',
  '성격상 강점': '-예의가 바르고 겸손한 성향.\n-맡겨진 일에 좋은 결과를 노력하지만 자신의 여가도 중요시 함.\n-관계 중심적으로 갈등과 싸움을 싫어해 중립적임.\n-이해와 배려심이 많아 친구들에게 인기가 많으며 친밀한 관계를 형성함.',
  '성격상 약점': '-소극적이며 내성적임.\n-게으른 면이 있을 수 있음.\n-소심해 스트레스를 많이 받음.\n-적극적이고 과감한 상황에 용기 부족.',
  '추천 직업': '공무원, 교사, 공무직, 금융업, 감리사, 세무사, IT개발자, 전문기능인, 직업상담사, 임상병리사, 재활치료사, 동물훈련사, 1차 산업, 환경운동가, 요식업'},
 '빨강': {'요약': '열정 카리스마 리더형',
  '성격상 강점': '-적극적이고 주도적임.\n-책임감이 강하고 승부욕이 있음\n-확신을 갖는 일에 소신껏 밀어부침.\n-실행력을 동반한 행동.파\n-뒤끝 없으며 과감함\n-자신감과 자의식이 강함.',
  '성격상 약점': '-성격이 급하고 충동적임.\n-이기적이고 자의식이 강해 자신의 뜻대로 안되면 화가 남.\n-원리원칙에서 벗어날 경우 상대를 불신함.\n-폭발적인 성격으로 감정 다스림이 필요함.',
  '추천 직업': '군인, 경찰, 판사, 검사, 검사, 흉부외과, 정형외과, 마케팅 전문가, 경영자, 사업가, 창업가, 정치인, 운동선수, 기업가, 모델'},
 '노랑': {'요약': '자존감이 높은 성장 마인드',
  '성격상 강점': '-새로운 것에 관심이 많고 호기심도 완성함.\n-변화를 두려워하지않으며 도전의식이 강함.\n-사교적인 성격으로 친구가 많으며 금새 사귀기도 함.\n-표현력이 풍부하고 마음이 따뜻해 인기가 많음.\n-긍정적인 성격으로 갈등에서도 빠르데 회복함.',
  '성격상 약점': '-다양성 추구로 금새 흥미를 잃음.\n-얇고넓은 지식이 있음.\n-질투를 쉽게 느낌.\n-이기고 싶은 욕구.\n-집중력, 진득함과 타인을 배려하는 태도가 필요함.',
  '추천 직업': '인플루언스, 전문 블로거, 에디터, 잡지, 광고, 언론, 방송인, 기상 캐스터, 기자, 리포터, 패션 디렉터, 머쳔다이저, 베이커리, 캐릭터 디자인'},
 '주황': {'요약': '활력 넘치는 낙천가 유형',
  '성격상 강점': '-재미, 흥미, 활기를 추구함.\n-외부 세계에 관심이 많으며 역동적인 에너지가 충만함.\n-창의성과 아이디어가 많아 친구들에게 인기가 많음.\n-순발력과 임기응변이 뛰어남.',
  '성격상 약점': '-집중력이 매우 짧고 산만함.\n-경솔하거나 무모한 태도가 있음.\n-책임감이 적어 맡은 일을 완수하지 못함.\n-자신만 주목받고 싶은 욕구가 있어 주목받는 사람에게 질투를 느낌.\n-끈기과 인내가 필요하며 계획을 세워 실천하는 책임감 필요.',
  '추천 직업': '연예인, 개그맨, 댄서, 마술사, 메이크업 아티스트, 인터넷 방송인, 응원단, 예술가, 영업, 마케팅, 세일즈맨, 홍보담당자, 레저활동 전문가, 여행, 숙박업, 스포츠 매니저, 운동선수'},
 '갈색': {'요약': '침착한 속 싶은 헌신가',
  '성격상 강점': '-마음이 따뜻하고 타인에게 도움이 되고 싶어하며 배려하는 성격.\n-마음에 든 사람과 가족에게 헌신함.\n-주어진 일에 책임감과 성실성이 있음.\n-인내와 끈기가 강함.',
  '성격상 약점': '·변화를 싫어하고 느림\n-타인을 과하게 배려해 손해를 봄.\n-고집스럽고 보수적인 성격.\n-촌스럽거나 답답해 보일 수 있음.\n-변화를 두려워하지말고 도전하는 자세 필요.',
  '추천 직업': '도서관 사서, 문서 작성가, 출판사 편집자, 고고학자, 간호사, 교사, 공무원, 비서, 사회복지사, 장애인 복지사, 상담사, 요리사, 세무사, 감리사, 농업계 연구원, \n가구 디자이너, 현모양처'},
 '검정': {'요약': '조용한 목표 추구형',
  '성격상 강점': '-타인에게 의존하지 않고 스스로 해결하는 유형.\n-자기 주도성이 강하며 조용히 목표를 세우고 실천해 나감.\n-자기 생각과 감정에 집중하며 자신만의 생활방식이 있음.\n-관심분야에 높은 집중력을 보임.',
  '성격상 약점': '-표현능력이 부족하며 사교성이 적음\n-관심분야 외 소극적인 태도.\n-타인의 행동이나 생각을 비판적으로 바라보는 경향이 있음.\n-자신의 생각에 대한 고집이 강함.\n-소통, 공감능력과 사교성 필요.',
  '추천 직업': '기업가, 정치인, 로비스트, 딜러, 연구원, 엔지니어, 과학자'},
 '파랑': {'요약': '스마트한 논리와 분석형',
  '성격상 강점': '-분석적이고 이성적임.\n-학문탐구적 성향.\n-따뜻한 마음으로 봉사에 대한 관심.\n-성실하며 계획적으로 맡겨진 일을 끈기있게 해 냄.\n-책임감이 강하며 타인에게 신뢰감을 줌.',
  '성격상 약점': '-신념이 확고해 고집이 세고 굽히려 하지 않음.\n-자기 주장을 펼쳐 독선적일 때가 있음.\n-규칙에 반하면 불편함을 느낌.\n-군중 속의 외로움을 느껴 심할 경우 우울감이 있을 수 있음.',
  '추천 직업': '조력가, 교수, 교사, 뇌과학자, 내과의사, 한의사, 연구원, 역사학자, 변호사, 회계사, 컨설턴트, 금융전문가, 공무원, 엔지니어, 경찰, 출판업, 수필가, 비평가, 물리학자, 수학자, 철학자, 사상가'},
 '하양': {'요약': '다재다능한 완벽 추구형',
  '성격상 강점': '-자신과 타인에게 엄격한 기준을 가지고 있으며 완벽을 추구함.\n-순수하고 정직함을 중요시하며 진실한 성향.\n-다양한 분야에 재능을 보이며 높은 목표를 가짐.\n-깔끔한 성격으로 정리정돈이 되어 있어야 마음이 놓임.',
  '성격상 약점': '-높은 목표와 완벽한 결과를 위해 지칠 수 있음.\n-실수를 용인하는 태도 필요.\n-문제해결 중심으로 감성과 공감 부족\n-협력과 조화가 필요.',
  '추천 직업': '검사, 판사, 법무사, 법학교수, 의사, 약사, 검안관, 보건복지 관련 공무원, 섬세한 기술의 엔지니어, 공학 기술자, 과학자, 경제학자, 금융전문가, 회계사'}}

# 홍길동의 가장 높은 점수를 받은 색깔 찾기
highest_score_color = student_scores_df.idxmax(axis=1).iloc[0]

# Streamlit 앱에서 사용할 CSS 정의
custom_css = """
    <style>
        .custom-box {
            border: 1px solid #f3f3f3;
            border-radius: 5px;
            background-color: #f3f3f3;
            padding: 10px;
            font-weight: bold;
        }
    </style>
"""
st.markdown(custom_css, unsafe_allow_html=True)
# 해당 색깔의 정보 추출
highest_score_color_info = color_info_dict_full[highest_score_color]
color_info = highest_score_color_info['요약']
# 색상 정보를 보다 구조화된 형태로 출력
st.markdown(f'<div class="custom-box">최고 점수 색상 : {highest_score_color}</div>', unsafe_allow_html=True)
# <br> 태그를 사용하여 공백 추가
st.markdown('<br>', unsafe_allow_html=True)
st.write(f'<div class="custom-box">요약 : {color_info}</div>',unsafe_allow_html=True)

# 성격상 강점 및 약점에서 줄바꿈 문자를 쉼표로 교체하여 출력

character_strengths = highest_score_color_info['성격상 강점']
character_weaknesses = highest_score_color_info['성격상 약점']


st.markdown('<br>', unsafe_allow_html=True)
st.write(f'<div class="custom-box">성격상 강점</div>',unsafe_allow_html=True)
st.text(f"{character_strengths}")

st.markdown('<br>', unsafe_allow_html=True)
st.write(f'<div class="custom-box">성격상 약점</div>',unsafe_allow_html=True)
st.text(f"{character_weaknesses}")

# 추천 직업 출력
st.markdown('<br>', unsafe_allow_html=True)
st.write(f'<div class="custom-box">추천 직업</div>',unsafe_allow_html=True)
st.text(f"{highest_score_color_info['추천 직업']}")

st.markdown("""
#### 컬러별 특성 설명
""")

color_info_1 = {
    'NO': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "구분": ['하양', '노랑', '주황', '분홍', '빨강', '초록', '파랑', '갈색', '보라', '검정'],
    "설명": ["완벽성과 다재다능, 민감하고 스트레스 취약, 정직과 청렴, 성실한 책임완수",
             "진취적인 성향, 밝고 명랑한 관계 지향, 탐구심, 질투심, 비판적, 책임감 부족",
             "활발하고 자유분방함, 자립적, 용감함, 도전적, 추진력, 집중력과 인내심 부족",
             "다정함, 친절함, 섬세하고 감성적, 예술성, 화합, 순응적, 불안감, 독립심 부족",
             "도전적, 자신감, 책임감, 솔직함, 의리, 과감함, 다양한 경험, 감정적, 충동성",
             "책임감과 성실함, 예의바른, 겸손함, 이해심, 평화적, 갈등 해결, 소심함, 소극적",
             "탐구적, 지성적, 분석력, 치밀함, 계획성, 책임감, 규칙과 질서, 고독, 우울, 독단",
             "안정감과 평온함, 인내, 헌신, 봉사, 내성적, 보수적, 융통성과 이해력 부족, 지루함",
             "신비롭고 매력적, 통찰력, 독창성, 예술, 자유로움, 자존심이 강함, 우월감, 고집",
             "독립성, 내성적, 깊은 생각, 통찰력, 많은 비밀, 현실적인, 고독, 우울, 내면을 숨김"
            ]
}

# DataFrame으로 변환
df_color = pd.DataFrame(color_info_1)

st.write(df_color.set_index('NO'))

# 'character' 시트의 Emotion 정보를 사용하여 문제 ID를 색깔별로 그룹화합니다.
emotion_groups_new = character_sheet[['Question', 'Emotion']].set_index('Question')['Emotion'].to_dict()

# 학생별로 각 색깔에 대한 합산 점수를 계산하기 위한 딕셔너리를 초기화합니다.
student_emotion_scores_new = {student: {color: 0 for color in set(emotion_groups_new.values())} for student in score_sheet_adjusted_proper_transposed.columns}
# 점수 데이터를 사용하여 각 문제에 대한 점수를 가져와 색깔 그룹에 맞춰 합산합니다.
for problem_id, scores in df_dict.items():
    emotion = emotion_groups_new.get(problem_id)
    for student, score in scores.items():
        student_emotion_scores_new[student][emotion] += score
        
student_emotion_scores_df_uploaded = pd.DataFrame(student_emotion_scores_new).T
# 각 학생별로 점수가 높은 순으로 색깔을 정렬합니다.
sorted_scores_df_uploaded_emotion = student_emotion_scores_df_uploaded.apply(lambda x: x.sort_values(ascending=False), axis=1)


#--------------------------------------------표시------------------------------------------------
st.markdown("""
### 2. 정서 행동 특성 검사 결과
""")

st.markdown("""
#### 정서 행동 특성 설명
""")

data = {
    'NO': [1, 2, 3, 4, 5, 6],
    '구분': ['책임감', '사교성', '자기표현', '자존감', '공감능력', '자기효능감'],
    '설명': [
        '맡아서 해야 할 임무나 의무를 중요하게 여기는 마음과 태도 \n\n\n\n',
        '타인과 원만하게 상호작용 하는 능력',
        '자신의 마음을 정직하게 표현 할 수 있는 사회적 능력',
        '자신을 소중하고 가치있는 존재로 여기는 마음',
        '타인의 감정을 이해하고 공유하는 능력',
        '어떤 상황에서 적절한 행동을 할 수 있다는 기대와 신념',
    ]
}

df = pd.DataFrame(data)
# 스트림릿에 표를 나타냅니다.
st.write(df.set_index('NO'))


responsibility = sorted_scores_df_uploaded_emotion['책임감'][selected_student]
sociability = sorted_scores_df_uploaded_emotion['사교성'][selected_student]
self_expression = sorted_scores_df_uploaded_emotion['자기표현'][selected_student]
self_esteem = sorted_scores_df_uploaded_emotion['자존감'][selected_student]
empathy = sorted_scores_df_uploaded_emotion['공감능력'][selected_student]
self_efficacy = sorted_scores_df_uploaded_emotion['자기효능감'][selected_student]

responsibility_ratio = responsibility / 35
responsibility_ratio = f"{responsibility_ratio * 100:.0f}%"

sociability_ratio = sociability / 45
sociability_ratio = f"{sociability_ratio * 100:.0f}%"

self_expression_ratio = self_expression / 35
self_expression_ratio = f"{self_expression_ratio * 100:.0f}%"

self_esteem_ratio =self_esteem / 30
self_esteem_ratio = f"{self_esteem_ratio * 100:.0f}%"

empathy_ratio =  empathy / 30
empathy_ratio = f"{empathy_ratio * 100:.0f}%"

self_efficacy_ratio = self_efficacy / 30
self_efficacy_ratio = f"{self_efficacy_ratio * 100:.0f}%"

# 홍길동의 각 항목별 점수
scores = {
    '책임감': responsibility,
    '사교성': sociability,
    '자기표현': self_expression,
    '자존감': self_esteem,
    '공감능력': empathy,
    '자기효능감': self_efficacy
}

# 각 항목별 점수 기준
score_criteria = {
    '책임감': {'높음': (26, 35), '보통': (16, 25), '낮음': (7, 15)},
    '사교성': {'높음': (33, 45), '보통': (21, 32), '낮음': (9, 20)},
    '자기표현': {'높음': (26, 35), '보통': (16, 25), '낮음': (7, 15)},
    '자존감': {'높음': (22, 30), '보통': (14, 21), '낮음': (6, 13)},
    '공감능력': {'높음': (22, 30), '보통': (14, 21), '낮음': (6, 13)},
    '자기효능감': {'높음': (22, 30), '보통': (14, 21), '낮음': (6, 13)}
}

st.markdown(f"<h4 style='text-align:left; color: black;'>{selected_student} 검사 결과 </h4>", unsafe_allow_html=True)
# 결과 데이터 초기화
data_emotion = {
    '구분': ['책임감', '사교성', '자기표현', '자존감', '공감능력', '자기효능감'],
    '높음': [],
    '보통': [],
    '낮음': [],
    '점수': [scores['책임감'], scores['사교성'], scores['자기표현'], scores['자존감'], scores['공감능력'], scores['자기효능감']],
    '비중': [responsibility_ratio, sociability_ratio, self_expression_ratio, self_esteem_ratio, empathy_ratio, self_efficacy_ratio],  # 비중은 여기서는 계산하지 않고 비워둡니다.
    '높음(기준)': ['35~26', '45~33', '35~26', '30~22', '30~22', '30~22'],
    '보통(기준)': ['25~16', '32~21', '25~16', '21~14', '21~14', '21~14'],
    '낮음(기준)': ['15~7', '20~9', '15~7', '13~6', '13~6', '13~6']
}

# 결과 데이터 초기화
data_emotion_2 = {
    'NO': [1, 2, 3, 4, 5, 6],
    '구분': ['책임감', '사교성', '자기표현', '자존감', '공감능력', '자기효능감'],
    '점수': [scores['책임감'], scores['사교성'], scores['자기표현'], scores['자존감'], scores['공감능력'], scores['자기효능감']],
    '비중': [responsibility_ratio, sociability_ratio, self_expression_ratio, self_esteem_ratio, empathy_ratio, self_efficacy_ratio],  # 비중은 여기서는 계산하지 않고 비워둡니다.

}
df_2 = pd.DataFrame(data_emotion_2)
# 각 항목별 점수를 기준에 따라 분류
for category, score in scores.items():
    for classification, (min_score, max_score) in score_criteria[category].items():
        if min_score <= score <= max_score:
            data_emotion[classification].append('O')
        else:
            data_emotion[classification].append('')

df = pd.DataFrame(data_emotion)
st.write(df.set_index('구분'))

score_ratio = {
     '책임감': responsibility_ratio,
     '사교성': sociability_ratio,
     '자기표현': self_expression_ratio,
     '자존감': self_esteem_ratio,
     '공감능력': empathy_ratio,
     '자기효능감': self_efficacy_ratio 
}

# 백분율 문자열을 실수로 변환
scores = [float(rate[:-1]) for rate in score_ratio.values()]

# 라벨
labels = list(score_ratio.keys())

# 각 라벨의 각도 계산
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()

# 그래프를 닫기 위해 시작점으로 돌아가는 점수와 각도 추가
scores += scores[:1]
angles += angles[:1]

# 방사형 그래프 생성
fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
ax.fill(angles, scores, alpha=0)
ax.plot(angles, scores, color='blue', linewidth=1)

# Annotate each point with its score
for label, angle, score in zip(labels, angles, scores):
    ax.text(angle, score, f"{int(score)}%", ha='center', va='center', fontsize=15, color='blue')

# 라벨 위치 설정
ax.set_thetagrids(np.degrees(angles[:-1]), labels)

# 원을 10%부터 100%까지 설정
yticks = np.linspace(10, 100, 10)  # 0.1부터 1.0까지 10개의 틱 생성
ax.set_yticks(yticks)  # y축 틱 설정

col1, col2, col3 = st.columns([5, 7, 2])

with col1:
    st.write(df_2.set_index('NO'))

with col2:
    st.pyplot(plt)

    
st.markdown(f"<br>",unsafe_allow_html=True)
st.markdown(f"<br>",unsafe_allow_html=True)
st.markdown("""
   ##### (유)Edupia 홍쌤 색채심리 연구소
""")
