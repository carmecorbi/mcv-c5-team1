import evaluate


class Metric:
    def __init__(self):
        # Initialize the Hugging Face metric objects for BLEU, ROUGE, and METEOR
        self.bleu = evaluate.load("bleu")
        self.rouge = evaluate.load("rouge")
        self.meteor = evaluate.load("meteor")

    def compute_bleu(self, predictions, references, max_order=1):
        """
        Computes BLEU score for a given set of predictions and references.
        max_order can be 1 or 2 for BLEU-1 and BLEU-2 respectively.
        """
        if max_order not in [1, 2]:
            raise ValueError("max_order should be 1 or 2 for BLEU-1 and BLEU-2.")
        
        # BLEU scores using Hugging Face evaluate library
        bleu_score = self.bleu.compute(predictions=predictions, references=references)
        return bleu_score['bleu']
    
    def compute_rouge(self, predictions, references):
        """
        Computes ROUGE-L score for the given predictions and references.
        """
        # ROUGE scores using Hugging Face evaluate library
        rouge_scores = self.rouge.compute(predictions=predictions, references=references)
        return rouge_scores['rougeL']
    
    def compute_meteor(self, predictions, references):
        """
        Computes METEOR score for the given predictions and references.
        """
        # METEOR scores using Hugging Face evaluate library
        meteor_score_value = self.meteor.compute(predictions=predictions, references=references)
        return meteor_score_value['meteor']
    
    def __call__(self, predictions, references, max_order=1):
        """
        Computes BLEU, ROUGE-L, and METEOR scores.
        This method can be called to compute all metrics at once.
        """
        bleu_score_1 = self.compute_bleu(predictions, references, max_order=1)  # BLEU-1
        bleu_score_2 = self.compute_bleu(predictions, references, max_order=2)  # BLEU-2
        rouge_L_score = self.compute_rouge(predictions, references)  # ROUGE-L
        meteor_score_value = self.compute_meteor(predictions, references)  # METEOR
        
        # Return all metrics in a dictionary
        return {
            'bleu1': bleu_score_1,
            'bleu2': bleu_score_2,
            'rougeL': rouge_L_score,
            'meteor': meteor_score_value
        }
