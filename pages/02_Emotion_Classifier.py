import streamlit as st
import time
import torch
import pandas as pd
import numpy as np
from transformers import BartForSequenceClassification, PreTrainedTokenizerFast

st.set_page_config(page_title = "Emotion", layout = "wide", initial_sidebar_state = "expanded")

HF_PATH = "yelim24/utterance_emotion_classification"

@st.cache_resource
def get_model():
    model = BartForSequenceClassification.from_pretrained(HF_PATH, num_labels=4, ignore_mismatched_sizes=True)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(HF_PATH)
    return EmotionClassifier(model=model, tokenizer=tokenizer)

class EmotionClassifier():
    def __init__(self,
                 model,
                 tokenizer,
                 device="cpu"):
        self.device = device
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.labels = ("분노", "슬픔", "중립", "행복")

    @torch.no_grad()
    def get_predict(self,
                    input_text: str):
        inputs = self.tokenizer(self.tokenizer.bos_token+input_text+self.tokenizer.eos_token,
                                return_tensors='pt',
                                truncation=True,
                                max_length=128,
                                pad_to_max_length=True,
                                add_special_tokens=True)

        # self.dl_model.to(self.device)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        output = self.model(input_ids=input_ids,
                            attention_mask=attention_mask)
        outputmap = torch.nn.Softmax()(output.logits[0]).detach().cpu().numpy().astype(np.float64).round(3).tolist()
        result = dict(zip(self.labels, outputmap))
        predict_index = torch.argmax(output.logits[0])
        selected_emo = self.labels[predict_index.item()]
        # result.update({"result": self.labels[predict_index.item()]})
        # print(f"EmotionClassification 동작 소요 시간 : {time.time() - total}")
        return result, selected_emo

def main():
    emotion_classifier = get_model()
    
    st.sidebar.title("Emotion classifier")
    st.sidebar.markdown("""어쩌구 저쩌구 페이지 설명""")

    st.sidebar.markdown("---")
    st.sidebar.caption("Made by [yelim kim](mailto:kyelim24@gmail.com)")
    st.markdown("""
        # Emotion classifier
        
        어쩌구 저쩌구 설명~~ 4가지 감정 분류 가능~~
        """)
    clf_form = st.form(key='emotion_clf_form')
    with clf_form:
        # st.markdown("## 아래 칸에 문장을 입력해주세요 👇")
        text = st.text_input("아래 칸에 문장을 입력해주세요 👇", #value="오늘 날씨가 너무 좋지 않아?", 
                            #  label_visibility="hidden",
                             placeholder="오늘 날씨가 너무 좋지 않아?",)
        submit = st.form_submit_button(label='결과 보기')
        
    if submit:
        if text:
            result, selected_emo = emotion_classifier.get_predict(input_text = text)
            _, mid, _ = st.columns(3)
            with mid:
                with st.spinner('분석 중입니다..🏃‍♂️..'):
                    time.sleep(3)
            col1, col2 = st.columns(2)
            col1.success(f"Emotion Predicted : {selected_emo}")
            col2.success(f"Emotion Predicted : {result[selected_emo]}")
            
            st.markdown("""### Classification Probability""")
            result_df = pd.DataFrame([result])
            st.dataframe(result_df, hide_index=True)
        else:
            st.warning('문장을 입력해주세요!', icon="⚠️")
            # st.toast('문장을 입력해주세요!', icon="⚠️")
        
        
if __name__ == "__main__":
    main()

