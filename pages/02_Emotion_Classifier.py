import streamlit as st
import torch
import pandas as pd
import numpy as np
from transformers import BartForSequenceClassification, PreTrainedTokenizerFast



# HF_PATH = "yelim24/utterance_emotion_classification"
# model = BartForSequenceClassification.from_pretrained(HF_PATH, num_labels=4, ignore_mismatched_sizes=True)
# tokenizer = PreTrainedTokenizerFast.from_pretrained(HF_PATH)

# EMOTION_MAP = ("분노", "슬픔", "중립", "행복")

# class EmotionClassifier():
#     def __init__(self,
#                  model,
#                  tokenizer,
#                  device="cpu"):
#         self.device = device
#         self.model = model.to(self.device)
#         self.tokenizer = tokenizer
#         self.labels = EMOTION_MAP

#     @torch.no_grad()
#     def get_predict(self,
#                     input_text: str):
#         inputs = self.tokenizer(self.tokenizer.bos_token+input_text+self.tokenizer.eos_token,
#                                 return_tensors='pt',
#                                 truncation=True,
#                                 max_length=128,
#                                 pad_to_max_length=True,
#                                 add_special_tokens=True)

#         # self.dl_model.to(self.device)
#         input_ids = inputs['input_ids'].to(self.device)
#         attention_mask = inputs['attention_mask'].to(self.device)

#         output = self.model(input_ids=input_ids,
#                             attention_mask=attention_mask)
#         outputmap = torch.nn.Softmax()(output.logits[0]).detach().cpu().numpy().astype(np.float64).round(3).tolist()
#         result = dict(zip(self.labels, outputmap))
#         predict_index = torch.argmax(output.logits[0])
#         result.update({"result": self.labels[predict_index.item()]})
#         # print(f"EmotionClassification 동작 소요 시간 : {time.time() - total}")
#         return result
    
# emotion_classifier = EmotionClassifier(model=model,
#                                        tokenizer=tokenizer)

# user = "너무너무 화가 나"
# emotion_classifier.get_predict(input_text = user)

def main():
    
    st.set_page_config(page_title = "Emotion", layout = "wide", initial_sidebar_state = "expanded")

    # st.sidebar.title("Emotion classifier")
    st.sidebar.markdown("""# Emotion classifier
                        어쩌구 저쩌구 페이지 설명""")

    st.sidebar.markdown("---")
    st.sidebar.caption("Made by [yelim kim](mailto:kyelim24@gmail.com)")
    
    st.markdown("""
    # Emotion classifier
    
    어쩌구 저쩌구 설명~~ 4가지 감정 분류 가능~~
    """)
    
    with st.form(key='emotion_clf_form'):
        text = st.text_area("아래 칸에 문장을 입력해주세요👇\n", help="예) 오늘 날씨가 너무 좋지 않아?.")
        submit = st.form_submit_button(label='결과 보기')
        
        if submit:
            st.write("Text:", text)
        
if __name__ == "__main__":
    main()

