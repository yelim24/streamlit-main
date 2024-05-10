import scipy as sp
import numpy as np
import shap
import torch

class text_classification_pipeline():
    '''
        모델의 판단 기준을 shap으로 시각화 하기위한 classification pipeline 객체 -> 미사용
    '''
    def __init__(self, model=None, tokenizer=None, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        # self.model.to(self.device)
        self.explainer = shap.Explainer(self.text_analysis,  # SHAP explainer 선언
                                        self.tokenizer,
                                        output_names=["분노", "슬픔", "중립", "행복"])

    @torch.no_grad()
    def __call__(self, text_list=None,):
        # 분석 대상 텍스트 입력 처리 매서드
        # explainer에서 함수 호출을 통해 계산하기 때문에 __call__ 매서드에서 모델이 계산
        tokenized = [self.tokenizer.encode(f"{self.tokenizer.bos_token}{x}{self.tokenizer.eos_token}",
                                           padding="max_length",
                                           max_length=128,
                                           truncation=True,) for x in text_list]
        # return self.model(input_ids=torch.tensor(tokenized).to(self.device))[0]
        return self.model(input_ids=torch.tensor(tokenized))[0]
    
    def text_analysis(self, text=None):
        # 분석 완료된 value 만 출력하는 매서드
        outputs = self(text).detach().cpu().numpy()
        max_index = np.argmax(outputs, 1)
        return sp.special.logit((np.exp(outputs).T/np.exp(outputs).sum(-1)).T)

    @torch.no_grad()
    def explain_text(self, text: str = None, index: int = None):
        # SHAP value를 통한 입력 text explainer 실행
        if len(self.tokenizer.encode(text)) < 2:
            text += text
        # print("bdi index : ",index)
        shap_values = self.explainer([text], fixed_context=1)
        # with open("shap_object.html","w") as f:
        #     f.write(shap.plots.text(shap_values=shap_values[0],display=False))
        #     f.close()
        # print(shap_values)
        # return {
        #     "토큰": shap_values.data[0].tolist(),
        #     "중요도": shap_values.values[0].squeeze().tolist()
        # } #shap.plots.text(shap_values=shap_values[0], display=False)
        return shap_values
    