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

# Streamlit 앱의 제목 설정
st.title('휴먼컬러/정서 행동 특성 검사 결과')

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
## 1. 휴먼컬러 결과
### ▶ 검사자별 성격 유형 <최고점 50~최하점 10>
""")

# 결과 표시
student_scores_df = get_sorted_scores(selected_student)
st.write(student_scores_df)

st.write("""
### 최고 점수와 최하 점수 차이 분석 기준
- **7점 미만의 차이:** 소극적인 참여 또는 진실성 결여 가능성이 있음
- **7점~14점 차이:** 다양성, 수용성이 높으나 주관성, 가치관 정립이 낮을 수 있음
- **15점 이상 차이:** 또렷한 성격적 특성을 가지고 있음

### 점수 배점별 해석 기준
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
        plt.plot(colors, scores, marker='o', linestyle='-', color='b')
    
        # Annotate each point with its score
        for i, score in enumerate(scores):
            plt.text(colors[i], score + 0.5, str(score), ha='center', va='bottom', fontsize=10)
    
        # Adding title and labels
        plt.title(f'Score Distribution for {selected_student}')
        plt.xlabel('Subject')
        plt.ylabel('Score')
        plt.grid(True)
    
        # Display the plot in Streamlit
        st.pyplot(plt)
        
if __name__ == "__main__":
    main()
color_info_dict_full = {'purple': {'요약': '독창성과 신비로운 보라',
  '성격상 강점': '자신을 특별하다고 여기며 독특한 매력을 인정받고 싶어함.\n정신적이고 신비로운 것에 끌리며 탐구력을 보임.\n믿음과 신뢰를 구축한 사람은 끝까지 함께하며 책임감을 가짐.\n변화와 안정감 두 개의 마음을 가짐.\n관계 중시.\n자신이 선택한 것에 책임감을 가짐.',
  '성격상 약점': '양가감정이 있어 변덕스럽거나 감정기복이 있음.\n정신적인 것을 너무 추구하면 고독, 불안, 우울이 있을 수 있음.\n정체성의 혼란이 있을 수 있음.\n타인과 구별되는 우월감을 가짐.',
  '추천 직업': '행위예술가, 연극배우, 수필가, 극작가, 시인, 종교인, 상담사, 치료사, 사진가, 건축가, 실내디자이너, 일러스트레이터, 문학 연구자, 사회학자'},
 'pink': {'요약': '상냥하며 매력적인 유형',
  '성격상 강점': '온화하고 따뜻한 성격의 소유자.\n타인의 감정에 대해 민감하며 공감능력이 뛰어남.\n감수성이 풍부하며 애교가 많음.\n타인이 나를 어떻게 생각하는지에 대한 관심이 많음.\n보호하거나 보호받고 싶어함.\n협조적이며 순응적인 성격.',
  '성격상 약점': '즉흥적 감정으로 표현하는 경우가 있어 신중함이 적음\n사랑, 관심 받고 싶은 마음이 커서 사람과의 관계에서 자주 스트레스를 받음\n독립심이 적고 의존적인 성향임\n허영심이 있기도 하며 무책임함이 있을 수 있음',
  '추천 직업': '예술가, 음악가, 공예가, 마술사, 댄서, 모델, 반려견 사업, 애완견 미용, 플로리스트, 호텔리어, 카페, 웨딩플래너, 메이크업아티스트, 간호사, 보육교사, \n어린이 관련 직업'},
 'green': {'요약': '예의바르며 성실한 안정형',
  '성격상 강점': '예의가 바르고 겸손한 성향\n맡겨진 일에 좋은 결과를 노력하지만 자신의 여가도 중요시 함\n관계 중심적으로 갈등과 싸움을 싫어해 중립적임\n이해와 배려심이 많아 친구들에게 인기가 많으며 친밀한 관계를 형성함',
  '성격상 약점': '소극적이며 내성적임\n게으른 면이 있을 수 있음\n소심해 스트레스를 많이 받음\n적극적이고 과감한 상황에 용기 부족',
  '추천 직업': '공무원, 교사, 공무직, 금융업, 감리사, 세무사, IT개발자, 전문기능인, 직업상담사, 임상병리사, 재활치료사, 동물훈련사, 1차 산업, 환경운동가, 요식업'},
 'red': {'요약': '열정 카리스마 리더형',
  '성격상 강점': '적극적이고 주도적임\n책임감이 강하고 승부욕이 있음\n확신을 갖는 일에 소신껏 밀어부침\n실행력을 동반한 행동파\n뒤끝 없으며 과감함\n자신감과 자의식이 강함',
  '성격상 약점': '성격이 급하고 충동적임\n이기적이고 자의식이 강해 자신의 뜻대로 안되면 화가 남\n원리원칙에서 벗어날 경우 상대를 불신함\n폭발적인 성격으로 감정 다스림이 필요함',
  '추천 직업': '군인, 경찰, 판사, 검사, 검사, 흉부외과, 정형외과, 마케팅 전문가, 경영자, 사업가, 창업가, 정치인, 운동선수, 기업가, 모델'},
 'yellow': {'요약': '자존감이 높은 성장 마인드',
  '성격상 강점': '새로운 것에 관심이 많고 호기심도 완성함\n변화를 두려워하지않으며 도전의식이 강함\n사교적인 성격으로 친구가 많으며 금새 사귀기도 함\n표현력이 풍부하고 마음이 따뜻해 인기가 많음\n긍정적인 성격으로 갈등에서도 빠르데 회복함',
  '성격상 약점': '다양성 추구로 금새 흥미를 잃음\n얇고넓은 지식이 있음\n질투를 쉽게 느낌\n이기고 싶은 욕구\n집중력, 진득함과 타인을 배려하는 태도가 필요함',
  '추천 직업': '인플루언스, 전문 블로거, 에디터, 잡지, 광고, 언론, 방송인, 기상 캐스터, 기자, 리포터, 패션 디렉터, 머쳔다이저, 베이커리, 캐릭터 디자인'},
 'orange': {'요약': '활력 넘치는 낙천가 유형',
  '성격상 강점': '재미, 흥미, 활기를 추구함\n외부 세계에 관심이 많으며 역동적인 에너지가 충만함\n창의성과 아이디어가 많아 친구들에게 인기가 많음\n순발력과 임기응변이 뛰어남',
  '성격상 약점': '집중력이 매우 짧고 산만함\n경솔하거나 무모한 태도가 있음\n책임감이 적어 맡은 일을 완수하지 못함\n자신만 주목받고 싶은 욕구가 있어 주목받는 사람에게 질투를 느낌\n끈기과 인내가 필요하며 계획을 세워 실천하는 책임감 필요',
  '추천 직업': '연예인, 개그맨, 댄서, 마술사, 메이크업 아티스트, 인터넷 방송인, 응원단, 예술가, 영업, 마케팅, 세일즈맨, 홍보담당자, 레저활동 전문가, 여행, 숙박업, \n스포츠 매니저, 운동선수'},
 'brown': {'요약': '침착한 속 싶은 헌신가',
  '성격상 강점': '마음이 따뜻하고 타인에게 도움이 되고 싶어하며 배려하는 성격\n마음에 든 사람과 가족에게 헌신함\n주어진 일에 책임감과 성실성이 있음\n인내와 끈기가 강함',
  '성격상 약점': '변화를 싫어하고 느림\n타인을 과하게 배려해 손해를 봄\n고집스럽고 보수적인 성격\n촌스럽거나 답답해 보일 수 있음\n변화를 두려워하지말고 도전하는 자세 필요',
  '추천 직업': '도서관 사서, 문서 작성가, 출판사 편집자, 고고학자, 간호사, 교사, 공무원, 비서, 사회복지사, 장애인 복지사, 상담사, 요리사, 세무사, 감리사, 농업계 연구원, \n가구 디자이너, 현모양처'},
 'black': {'요약': '조용한 목표 추구형',
  '성격상 강점': '타인에게 의존하지 않고 스스로 해결하는 유형\n자기 주도성이 강하며 조용히 목표를 세우고 실천해 나감\n자기 생각과 감정에 집중하며 자신만의 생활방식이 있음\n관심분야에 높은 집중력을 보임',
  '성격상 약점': '표현능력이 부족하며 사교성이 적음\n관심분야 외 소극적인 태도\n타인의 행동이나 생각을 비판적으로 바라보는 경향이 있음\n자신의 생각에 대한 고집이 강함\n소통, 공감능력과 사교성 필요',
  '추천 직업': '기업가, 정치인, 로비스트, 딜러, 연구원, 엔지니어, 과학자'},
 'blue': {'요약': '스마트한 논리와 분석형',
  '성격상 강점': '분석적이고 이성적임\n학문탐구적 성향\n따뜻한 마음으로 봉사에 대한 관심\n성실하며 계획적으로 맡겨진 일을 끈기있게 해 냄\n책임감이 강하며 타인에게 신뢰감을 줌',
  '성격상 약점': '신념이 확고해 고집이 세고 굽히려 하지 않음\n자기 주장을 펼쳐 독선적일 때가 있음\n규칙에 반하면 불편함을 느낌\n군중 속의 외로움을 느껴 심할 경우 우울감이 있을 수 있음',
  '추천 직업': '조력가, 교수, 교사, 뇌과학자, 내과의사, 한의사, 연구원, 역사학자, 변호사, 회계사, 컨설턴트, 금융전문가, 공무원, 엔지니어, 경찰, 출판업, 수필가, 비평가, \n물리학자, 수학자, 철학자, 사상가'},
 'white': {'요약': '다재다능한 완벽 추구형',
  '성격상 강점': '자신과 타인에게 엄격한 기준을 가지고 있으며 완벽을 추구함\n순수하고 정직함을 중요시하며 진실한 성향\n다양한 분야에 재능을 보이며 높은 목표를 가짐\n깔끔한 성격으로 정리정돈이 되어 있어야 마음이 놓임',
  '성격상 약점': '높은 목표와 완벽한 결과를 위해 지칠 수 있음\n실수를 용인하는 태도 필요\n문제해결 중심으로 감성과 공감 부족\n협력과 조화가 필요',
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

character_strengths = highest_score_color_info['성격상 강점'].replace('\n', ', ')
character_weaknesses = highest_score_color_info['성격상 약점'].replace('\n', ', ')


st.markdown('<br>', unsafe_allow_html=True)
st.write(f'<div class="custom-box">성격상 강점</div>',unsafe_allow_html=True)
st.write(f"{character_strengths}")

st.markdown('<br>', unsafe_allow_html=True)
st.write(f'<div class="custom-box">성격상 약점</div>',unsafe_allow_html=True)
st.write(f"{character_weaknesses}")

# 추천 직업 출력
st.markdown('<br>', unsafe_allow_html=True)
st.write(f'<div class="custom-box">추천 직업</div>',unsafe_allow_html=True)
st.write(f"{highest_score_color_info['추천 직업']}")


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

st.markdown("""
## 2. 정서 행동 특성 검사 결과
""")
# 데이터 프레임을 생성합니다.
data = {
    'NO': [1, 2, 3, 4, 5, 6],
    '구분': ['공감능력', '자기효능감', '책임감', '사교성', '자기표현능력', '자존감'],
    '설명': [
        '타인의 감정을 이해하고 공유하는 능력',
        '어떤 상황에서 적절한 행동을 할 수 있다는 기대와 신념',
        '맡아서 해야 할 임무나 의무를 중요하게 여기는 마음과 태도',
        '타인과 원만하게 상호작용 하는 능력',
        '자신의 마음을 정직하게 표현 할 수 있는 사회적 능력',
        '자신을 소중하고 가치있는 존재로 여기는 마음'
    ]
}

df = pd.DataFrame(data)
# 스트림릿에 표를 나타냅니다.
st.table(df.set_index('NO'))
st.write("(최고점 50점~최하점 10점)")

# 선택된 학생의 점수를 DataFrame 형태로 표시
def get_sorted_scores_df_by_input(student_name):
    if student_name in sorted_scores_df_uploaded_emotion.index:
        student_scores = sorted_scores_df_uploaded_emotion.loc[student_name].sort_values(ascending=False)
        return pd.DataFrame(student_scores).T
    else:
        return pd.DataFrame({"Error": ["Student name not found."]})



# 결과 표시
student_scores_df_emotion = get_sorted_scores_df_by_input(selected_student)
student_scores_df_emotion.drop(student_scores_df_emotion.columns[0], axis=1, inplace=True)
student_scores_df_emotion_transposed = student_scores_df_emotion.transpose()


labels = student_scores_df_emotion.columns.values
scores = student_scores_df_emotion.iloc[0].values
num_vars = len(labels)

# 각 축의 각도 계산
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
# 루프 완성
scores = np.concatenate((scores, [scores[0]]))
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.fill(angles, scores, color='red', alpha=0.25)
ax.plot(angles, scores, color='red', linewidth=2)

# 각 점수 라벨 그리기
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=20)

# 각 점수에 대해 주석 달기
for label, angle, score in zip(labels, angles, scores):
    ax.text(angle, score, str(score), ha='center', va='center', fontsize=20, color='blue')

ax.set_yticklabels([])

# Streamlit에 그래프 표시
col1, col2, col3 = st.columns([3, 7, 3]) # 화면을 세 부분으로 나눔

with col1: 
    st.write(student_scores_df_emotion_transposed )

with col2: # 가운데 컬럼에 그래프를 표시
    st.pyplot(fig)

