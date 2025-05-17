import json
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

from utils.util import ptb_tokenize
import fire
import pickle
import yaml
import torch
from torchvision import transforms

from datasets.flickr import Flickr8kDataset
from utils.metrics import bleu_score_fn
from utils.utils_torch import words_from_tensors_fn
from models import Vit_Captioner
from models import Res_Captioner
import random
import numpy as np
from tqdm import tqdm

class Runner(object):
    """Main class to run experiments"""
    def __init__(self, seed=1):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        self.device = torch.device(device)

    def evaluate_model(self, data_loader, model, bleu_score_fn,
        tensor_to_word_fn, word2idx, sample_method, desc='', return_output=False):
        bleu = [0.0] * 5
        model.eval()
        references = []
        predictions = []
        imgids = []
        t = tqdm(iter(data_loader), desc=f'{desc}', leave=False)
        for batch_idx, batch in enumerate(t):
            images, captions, lengths, imgid_batch = batch
            images = images.to(self.device)
            outputs = tensor_to_word_fn(model.sample(
                images, startseq_idx=word2idx['<start>'],
                method=sample_method).cpu().numpy())
            references.extend(captions)
            predictions.extend(outputs)
            imgids.extend(imgid_batch)
            t.set_postfix({
                'batch': batch_idx,
            }, refresh=True)
        t.close()
        for i in (1, 2, 3, 4):
            bleu[i] = bleu_score_fn(reference_corpus=references,
                                    candidate_corpus=predictions, n=i)
        references = [
           [" ".join(cap) for cap in caption] for caption in references
        ]
        predictions = [
            " ".join(caption) for caption in predictions
        ]
        return (bleu, references, predictions, imgids) if return_output else bleu

    def evaluate(self, config_file, **kwargs):
        with open(config_file) as reader:
            config = yaml.load(reader, Loader=yaml.FullLoader)
        args = dict(config, **kwargs)

        vocab_set = pickle.load(open(args['vocab_path'], "rb"))
        # Load the same dataset: flick8k
        test_set = Flickr8kDataset(dataset_base_path="data/flickr8k",
                                    dist='test', vocab_set=vocab_set,
                                    return_type='corpus',
                                    load_img_to_memory=False)
        vocab, word2idx, idx2word, max_len = vocab_set
        vocab_size = len(vocab)

        eval_transformations = transforms.Compose([
            transforms.Resize(args["image_len"]),  # smaller edge of image resized to img_len
            transforms.CenterCrop(args["image_len"]),  # get img_lenximg_len crop from random location
            transforms.ToTensor(),  # convert the PIL Image to a tensor
            transforms.Normalize((0.485, 0.456, 0.406),  # normalize image for pre-trained model
                                    (0.229, 0.224, 0.225))
        ])

        test_set.transformations = eval_transformations
        eval_collate_fn = lambda batch: (
            torch.stack([x[0] for x in batch]),
            [x[1] for x in batch], [x[2] for x in batch], [x[3] for x in batch])
        test_loader = torch.utils.data.DataLoader(test_set,
            batch_size=1, shuffle=False, collate_fn=eval_collate_fn)
            
        if args['model'] == 'resnet':
            model = Res_Captioner(encoded_image_size=14,
                                encoder_dim=2048, 
                                attention_dim=args['attention_dim'],
                                embed_dim=args['embedding_dim'],
                                decoder_dim=args['decoder_size'], 
                                vocab_size=vocab_size).to(self.device)

        
        elif args['model'] == 'vit':
            model = Vit_Captioner(encoded_image_size=14,
                                encoder_dim=768,
                                attention_dim=args['attention_dim'],
                                embed_dim=args['embedding_dim'],
                                decoder_dim=args['decoder_size'],
                                vocab_size=vocab_size).to(self.device)

        # model_path = os.path.join(args["outputpath"],
        #     f"{args['model']}_b{args['train_args']['batch_size']}_"
        #     f"emd{args['embedding_dim']}")

        # 这里要确定模型路径
        model_path = args["test_result_path"]
        state = torch.load(args["test_model_path"], map_location="cpu")
        model.load_state_dict(state["state_dict"])
        model = model.to(self.device)

        corpus_bleu_score_fn = bleu_score_fn(4, 'corpus')
        tensor_to_word_fn = words_from_tensors_fn(idx2word=idx2word)

        with torch.no_grad():
            model.eval()
            test_bleu, references, predictions, imgids = self.evaluate_model(
                desc=f'Test: ', model=model, bleu_score_fn=corpus_bleu_score_fn,
                tensor_to_word_fn=tensor_to_word_fn, data_loader=test_loader,
                sample_method=args['sample_method'], word2idx=word2idx,
                return_output=True)
            key_to_pred = {}
            key_to_refs = {}
            output_pred = []
            for imgid, pred, refs in zip(imgids, predictions, references):
                key_to_pred[imgid] = [pred,]
                key_to_refs[imgid] = refs
                output_pred.append({
                    "img_id": imgid,
                    "prediction": [pred,]
                })

            key_to_refs = ptb_tokenize(key_to_refs)
            key_to_pred = ptb_tokenize(key_to_pred)
            scorers = [Bleu(n=4), Rouge(), Meteor(), Cider(), Spice()]
            output = {"SPIDEr": 0}
            with open(f"{model_path}_scores.txt", "w") as writer:
                for scorer in scorers:
                    score, scores = scorer.compute_score(key_to_refs, key_to_pred)
                    method = scorer.method()
                    output[method] = score
                    if method == "Bleu":
                        for n in range(4):
                            print("Bleu-{}: {:.3f}".format(n + 1, score[n]), file=writer)
                    else:
                        print(f"{method}: {score:.3f}", file=writer)
                    if method in ["CIDEr", "SPICE"]:
                        output["SPIDEr"] += score
                output["SPIDEr"] /= 2
                print(f"SPIDEr: {output['SPIDEr']:.3f}", file=writer)

            json.dump(output_pred, open(f"{model_path}_predictions.json", "w"), indent=4)

if __name__ == "__main__":
    fire.Fire(Runner)