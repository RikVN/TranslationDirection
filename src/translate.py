#!/usr/bin/env python

'''Translate a given set of sentences to a target language using open-source HuggingFace models'''

import sys
import argparse
import time
import datetime
from transformers import pipeline
from create_translation_dataset import write_to_file, read_sents
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True,
                        help="Huggingface ID of the model. Can use shortened IDs - see below")
    parser.add_argument("-s", "--sent_file", required=True, type=str,
                        help="Predict on these sentences, not tokenized/processed yet")
    parser.add_argument("-b", "--batch_size", default=8, type=int,
                        help="Batch size during parsing")
    parser.add_argument("-ml", "--max_length", default=256, type=int,
                        help="Max length of input in terms of tokens")
    parser.add_argument("-o", "--output_file", type=str, required=True,
                        help="Output file to write translation to")
    parser.add_argument("-tf", "--take_first", action="store_true",
                        help="Take first column in tabbed file to translate")
    # IMPORTANT: the NLLB languages IDs are weird and do not use the ISO codes
    # https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200
    # e.g. Dutch is nld_Latn
    parser.add_argument("-sl", "--src_lang", type=str, required=True,
                        help="Source language of text")
    parser.add_argument("-tl", "--tgt_lang", type=str, required=True,
                        help="Target language of text")
    args = parser.parse_args()
    return args


def short_model_ids(model, src_lang, tgt_lang):
    '''Resolve a short model ID to actual model'''
    dic = {}
    # If we specify just opus, check src/tgt lang for the specific model
    # Basically check whether source or target is English
    eng_ids = ['en', 'eng', 'eng_Latn']
    if model == "opus":
        if src_lang in eng_ids:
            dic["opus"] = "Helsinki-NLP/opus-mt-en-mul"
        elif tgt_lang in eng_ids:
            dic["opus"] = "Helsinki-NLP/opus-mt-mul-en"
        else:
            raise ValueError("If the model is specified as OPUS, either src or tgt needs to be English")

    # Opus MT specific multi-lingual models
    dic["op-mul-en"] = "Helsinki-NLP/opus-mt-mul-en"
    dic["opus-mul-en"] = "Helsinki-NLP/opus-mt-mul-en"
    dic["mul-en"] = "Helsinki-NLP/opus-mt-mul-en"
    dic["op-en-mul"] = "Helsinki-NLP/opus-mt-en-mul"
    dic["opus-en-mul"] = "Helsinki-NLP/opus-mt-en-mul"
    dic["en-mul"] = "Helsinki-NLP/opus-mt-en-mul"
    # Large NLLB model
    dic["nllb-large"] = "facebook/nllb-200-3.3B"
    dic["nllb-3"] = "facebook/nllb-200-3.3B"
    dic["nllb3"] = "facebook/nllb-200-3.3B"
    dic["nllb3.3"] = "facebook/nllb-200-3.3B"
    dic["nllb3.3B"] = "facebook/nllb-200-3.3B"
    # Smaller NLLB model
    dic["nllb"] = "facebook/nllb-200-distilled-600M"
    dic["nllb-600"] = "facebook/nllb-200-distilled-600M"
    dic["nllb600"] = "facebook/nllb-200-distilled-600M"
    dic["nllb-600M"] = "facebook/nllb-200-distilled-600M"
    # Small M2M model of Facebook
    dic["m2m"] = "facebook/m2m100_418M"
    dic["m2m100"] = "facebook/m2m100_418M"
    dic["m2m100-small"] = "facebook/m2m100_418M"
    # Medium M2M model of Facebook
    dic["m2m-med"] = "facebook/m2m100_1.2B"
    dic["m2m-1.2"] = "facebook/m2m100_1.2B"
    dic["m2m-1.2B"] = "facebook/m2m100_1.2B"

    # Rewrite model if we find it
    if model in dic:
        print(f"INFO: use {dic[model]} for id {model}")
        return dic[model]
    return model


def rewrite_lang_code(lang, model):
    '''Rewrite the 2-letter ISO language codes to the longer language codes of facebook/OPUS'''
    dic = {}
    if 'nllb' in model.lower():
        dic['bg'] = 'bul_Cyrl'
        dic['cs'] = 'ces_Latn'
        dic['da'] = 'dan_Latn'
        dic['de'] = 'deu_Latn'
        dic['el'] = 'ell_Grek'
        dic['en'] = 'eng_Latn'
        dic['es'] = 'spa_Latn'
        dic['et'] = 'est_Latn'
        dic['fi'] = 'fin_Latn'
        dic['fr'] = 'fra_Latn'
        dic['hu'] = 'hun_Latn'
        dic['it'] = 'ita_Latn'
        dic['lt'] = 'lit_Latn'
        dic['lv'] = 'lvs_Latn'
        dic['nl'] = 'nld_Latn'
        dic['pl'] = 'pol_Lat'
        dic['pt'] = 'por_Latn'
        dic['ro'] = 'ron_Latn'
        dic['sk'] = 'slk_Latn'
        dic['sl'] = 'slv_Latn'
        dic['sv'] = 'swe_Latn'
        dic['tr'] = 'tur_Latn'
        dic['is'] = 'isl_Latn'
        dic['hr'] = 'hrv_Latn'
        # Maybe we specified the language code directly, then it's fine
        # Raise a warning though so we check
        if lang not in dic:
            print (f"WARNING: using language code {lang} directly for NLLB")
            ret = lang
        else:
            # Rewrite the language code for NLLB
            print(f"INFO: rewrite lang code {lang} to {dic[lang]}")
            ret = dic[lang]
    elif 'opus' in model.lower():
        dic['bg'] = 'bul'
        dic['cs'] = 'ces'
        dic['da'] = 'dan'
        dic['de'] = 'deu'
        dic['el'] = 'ell'
        dic['en'] = 'eng'
        dic['es'] = 'spa'
        dic['et'] = 'est'
        dic['fi'] = 'fin'
        dic['fr'] = 'fra'
        dic['hu'] = 'hun'
        dic['it'] = 'ita'
        dic['lt'] = 'lit'
        dic['lv'] = 'lav'
        dic['nl'] = 'nld'
        dic['pl'] = 'pol'
        dic['pt'] = 'por'
        dic['ro'] = 'ron'
        # Slovak is nowhere to be found in the OPUS models, I don't know why
        # Probably because it is similar to czech, so just use that model then
        dic['sk'] = 'ces'
        dic['sl'] = 'slv'
        dic['sv'] = 'swe'
        dic['is'] = 'isl'
        dic['tr'] = 'tur'
        dic['hr'] = 'hrv'
        # Maybe we specified the language code directly, then it's fine
        # Raise a warning though so we check
        if lang not in dic:
            print (f"WARNING: using language code {lang} directly for OPUS")
            ret = lang
        else:
            # Rewrite the language code for NLLB
            print(f"INFO: rewrite lang code {lang} to {dic[lang]}")
            ret = dic[lang]
    else:
        ret = lang
    return ret


def add_lang_token(sents, tgt_lang):
    '''Add the language token to the actual text for the OPUS model'''
    return [">>" + tgt_lang + "<< " + sent for sent in sents]


def main():
    '''Main function to translate an input file with an open source MT system'''
    args = create_arg_parser()

    # Rewrite model ID - makes things easier when using cmd line arguments
    model_id = short_model_ids(args.model, args.src_lang, args.tgt_lang)

    # Rewrite NLLB/OPUS language codes
    src_lang = rewrite_lang_code(args.src_lang, model_id)
    tgt_lang = rewrite_lang_code(args.tgt_lang, model_id)

    # Read in sentences to translate
    sents = read_sents(args.sent_file)

    # Maybe take only the first column
    if args.take_first:
        sents = [x.split('\t')[0].strip() for x in sents]

    # NOTE: the OPUS models need an actual token which specifies the target language
    # E.g. if we want to translate to Dutch, we have to add >>nld<< to the text
    # We can do this automatically using the specified tgt_lang
    if 'opus' in model_id.lower():
        sents = add_lang_token(sents, tgt_lang)

    # Very simple translation pipeline - device 0 means first GPU (default is -1)
    translator = pipeline("translation", model=model_id, device=0)
    print (f"INFO: start {src_lang}-{tgt_lang} translation processs for {len(sents)} sents with a max length of {args.max_length}")
    start_time = time.time()
    output = translator(sents, src_lang=src_lang, tgt_lang=tgt_lang, clean_up_tokenization_spaces=True, max_length=args.max_length)
    print(f"Generating translations took: {str(datetime.timedelta(seconds=time.time()-start_time))}\n\n")

    # Take just the sentences and write them to the output file
    sents = [x["translation_text"] for x in output]
    write_to_file(sents, args.output_file)


if __name__ == '__main__':
    # For logging purposes
    print("Generated by command:\npython", " ".join(sys.argv)+'\n')
    main()
