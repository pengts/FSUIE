from utils import IEDataset, tqdm, get_bool_ids_greater_than, logger
from transformers import BertTokenizerFast
from sparse_attn_model import UIE
# from model import UIE
import torch
from torch.utils.data import DataLoader
# from uie_predictor import UIEPredictor
import json
import argparse


def compute_metric(total_correct,total_label,total_pred):
    precision = total_correct / total_pred if total_correct else 0.0
    recall=total_correct/total_label if total_correct else 0.0
    f1=(2 * (precision * recall) / (precision + recall)) if total_correct else 0.0
    # print("P:",precision)
    # print("R:",recall)
    # print("f1:",f1)
    return precision,recall,f1

def get_span_for_eval(start_list,end_list):
    text_span=[]
    if len(start_list)==0 or len(end_list)==0:
        return[]
    i=0
    j=0
    while i<len(start_list)and j<len(end_list):
        if start_list[i]<=end_list[j]:
            text_span.append([start_list[i],end_list[j]])
            i+=1
        else:
            j+=1
    return text_span

def do_eval(model,test_data_loader,base_weight=0,fsl_weight=0.01,limit=0.5):
    model.eval()
    total_correct = 0
    total_label = 0
    total_pred = 0
    total_loss=0
    total_num=0


    for batch in tqdm(test_data_loader, desc="evaluating model"):
        input_ids, token_type_ids, att_mask, start_ids, end_ids = batch
        b=input_ids.size()[0]
        total_num+=b
        if args.device == 'gpu':
            input_ids = input_ids.cuda()
            # token_type_ids = token_type_ids.cuda()
            att_mask = att_mask.cuda()
            start_ids = start_ids.cuda()
            end_ids = end_ids.cuda()
        with torch.no_grad():
            outputs = model(input_ids=input_ids,
                            # token_type_ids=token_type_ids,
                            attention_mask=att_mask,
                            start_positions=start_ids,
                            end_positions=end_ids)
            total_loss+= base_weight * outputs.loss_base +  fsl_weight * outputs.loss_fsl
            start_ids = start_ids.float()
            end_ids = end_ids.float()
            start_pred = (outputs.start_prob > limit).float()
            end_pred = (outputs.end_prob > limit).float()

        for i in range(b):
            start_ids_list = torch.nonzero(start_ids[i])
            end_ids_list = torch.nonzero(end_ids[i])
            ids_spans = get_span_for_eval(start_ids_list, end_ids_list)

            start_pred_list = torch.nonzero(start_pred[i])
            end_pred_list = torch.nonzero(end_pred[i])
            pred_spans = get_span_for_eval(start_pred_list, end_pred_list)

            total_label += len(ids_spans)
            total_pred += len(pred_spans)

            for span in pred_spans:
                if span in ids_spans:
                    total_correct += 1
    avg_loss=total_loss.item()/total_num.item()
    total_precision, total_recall, total_f1 = compute_metric(total_correct.item(), total_label.item(),total_pred.item())
    model.train()
    return avg_loss,total_precision, total_recall, total_f1
    

def prepare():
    model = UIE.from_pretrained(args.model_path)
    tokenizer = BertTokenizerFast.from_pretrained(args.model_path)
    test_ds = IEDataset(args.test_path, tokenizer=tokenizer,
                        max_seq_len=args.max_seq_len)
    test_data_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    loss,precision, recall, f1=do_eval(model,test_data_loader)

    logger.info("Evaluation precision: %.5f, recall: %.5f, F1: %.5f" %
                (precision, recall, f1))

if __name__ == "__main__":
    # yapf: disable
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model", type=str, default='./NER_ACE05_FSUIE/model_best',
                        help="The path of saved model that you want to load.")
    parser.add_argument("-t", "--test_path", type=str, default='test.txt',
                        help="The path of test set.")
    parser.add_argument("-b", "--batch_size", type=int, default=32,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--max_seq_len", type=int, default=512,
                        help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--tokenizer", type=str,
                        help="Select the pretrained tokenizer.")
    args = parser.parse_args()

    prepare()
   

   



