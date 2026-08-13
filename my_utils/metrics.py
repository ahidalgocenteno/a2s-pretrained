import os
import tempfile
import uuid
import subprocess
from music21 import converter as converterm21
from pyMV2H.utils.mv2h import MV2H
from pyMV2H.metrics.mv2h import mv2h
from pyMV2H.utils.music import Music
from pyMV2H.converter.midi_converter import MidiConverter as Converter

from data_utils.encoding_convertions import VOICE_CHANGE_TOKEN, STEP_CHANGE_TOKEN, krnParser


def compute_metrics(y_true, y_pred, stage="val"):
    metrics = {}
    print(f"Computing edit distance metrics...")
    metrics = compute_ed_metrics(y_true=y_true, y_pred=y_pred)

    if stage == "test":
        # ################################# MV2H:
        print("Computing MV2H metrics... ")
        mv2h_dict = compute_mv2h_metrics(y_true=y_true, y_pred=y_pred)
        metrics.update(mv2h_dict)

        ################################# MUSTER:
        print("Computing MUSTER metrics...")
        muster_dict = compute_muster_metrics(y_true=y_true, y_pred=y_pred)
        metrics.update(muster_dict)

    print("Done!")
    return metrics


#################################################################### SYM-ER AND SEQ-ER:


def compute_ed_metrics(y_true, y_pred):
    def levenshtein(a, b):
        n, m = len(a), len(b)

        if n > m:
            a, b = b, a
            n, m = m, n

        current = range(n + 1)
        for i in range(1, m + 1):
            previous, current = current, [i] + [0] * n
            for j in range(1, n + 1):
                add, delete = previous[j] + 1, current[j - 1] + 1
                change = previous[j - 1]
                if a[j - 1] != b[i - 1]:
                    change = change + 1
                current[j] = min(add, delete, change)

        return current[n]

    ed_acc = 0
    length_acc = 0
    label_acc = 0
    for t, h in zip(y_true, y_pred):
        ed = levenshtein(t, h)
        ed_acc += ed
        length_acc += len(t)
        if ed > 0:
            label_acc += 1

    return {
        "sym-er": 100.0 * ed_acc / length_acc,
        "seq-er": 100.0 * label_acc / len(y_pred),
    }


############################### KERN FILE HANDLING FOR MV2H AND MUSTER:

def get_number_of_voices(kern_seq):
    num_voices = 0
    for token in kern_seq:
        if token == VOICE_CHANGE_TOKEN:
            continue
        if token == STEP_CHANGE_TOKEN:
            break
        num_voices += 1
    return num_voices

def create_kern_file(kern_seq, parser, num_voices, tmpdir):
    out_file = os.path.join(tmpdir, f"{uuid.uuid4()}.krn")

    parser.detokenize_to_string(
        token_sequence=kern_seq,
        n_voices=num_voices, 
        write_path=out_file
    )

    return out_file

def divide_voice(in_file, it_voice, tmpdir):
    out_file = os.path.join(tmpdir, f"{uuid.uuid4()}.krn")

    with open(in_file) as fin:
        lines = fin.readlines()

    voice = [u.split("\t")[it_voice].strip() for u in lines]

    with open(out_file, "w") as fout:
        for token in voice:
            fout.write(token + "\n")

    return out_file

#################################################################### MV2H:

def compute_mv2h_metrics(y_true, y_pred):

    # FILE CONVERSION
    
    kern_parser = krnParser()

    def krn2midi(in_file, tmpdir):
        midi_path = os.path.join(tmpdir, f"{uuid.uuid4()}.mid")
        converterm21.parse(in_file).write("midi", fp=midi_path)
        return midi_path
    
    def midi2txt(midi_file, tmpdir):
        txt_path = os.path.join(tmpdir, f"{uuid.uuid4()}.txt")
        converter = Converter(file=midi_file, output=txt_path)
        converter.convert_file()

        with open(txt_path, "r") as fin:
            lines = [u.replace(".0", "") for u in fin]

        with open(txt_path, "w") as fout:
            fout.writelines(lines)

        return txt_path


    # EVALUATIONS
    
    def eval_as_polyphonic(true_kern, pred_kern, tmpdir):
        ref_midi = krn2midi(true_kern, tmpdir)
        hyp_midi = krn2midi(pred_kern, tmpdir)

        ref_txt = midi2txt(ref_midi, tmpdir)
        hyp_txt = midi2txt(hyp_midi, tmpdir)

        try:
            return mv2h(
                Music.from_file(ref_txt),
                Music.from_file(hyp_txt),
            )
        except Exception:
            return MV2H(0, 0, 0, 0, 0)
        
    def eval_as_monophonic(true_kern, pred_kern, num_voices, tmpdir):
        global_res = MV2H(0, 0, 0, 0, 0)

        for it_voice in range(num_voices):
            true_voice = divide_voice(true_kern, it_voice, tmpdir)
            pred_voice = divide_voice(pred_kern, it_voice, tmpdir)

            ref_midi = krn2midi(true_voice, tmpdir)
            hyp_midi = krn2midi(pred_voice, tmpdir)

            ref_txt = midi2txt(ref_midi, tmpdir)
            hyp_txt = midi2txt(hyp_midi, tmpdir)

            try:
                res = mv2h(
                    Music.from_file(ref_txt),
                    Music.from_file(hyp_txt),
                )
                global_res.__multi_pitch__ += res.multi_pitch
                global_res.__voice__ += res.voice
                global_res.__meter__ += res.meter
                global_res.__harmony__ += res.harmony
                global_res.__note_value__ += res.note_value
            except Exception:
                pass

        for attr in ("__multi_pitch__", "__voice__", "__meter__", "__harmony__", "__note_value__"):
            setattr(global_res, attr, getattr(global_res, attr) / num_voices)

        return global_res

    # COMPUTATION OVER THE BATCH #######################################

    MV2H_global = MV2H(0, 0, 0, 0, 0)

    for t, h in zip(y_true, y_pred):
        with tempfile.TemporaryDirectory() as tmpdir:
            num_voices = get_number_of_voices(t)

            true_kern = create_kern_file(t, kern_parser, num_voices, tmpdir)
            pred_kern = create_kern_file(h, kern_parser, num_voices, tmpdir)

            try:
                converterm21.parse(pred_kern).write("midi")
                res = eval_as_polyphonic(true_kern, pred_kern, tmpdir)
            except Exception:
                res = eval_as_monophonic(true_kern, pred_kern, num_voices, tmpdir)

            MV2H_global.__multi_pitch__ += res.multi_pitch
            MV2H_global.__voice__ += res.voice
            MV2H_global.__meter__ += res.meter
            MV2H_global.__harmony__ += res.harmony
            MV2H_global.__note_value__ += res.note_value

    for attr in ("__multi_pitch__", "__voice__", "__meter__", "__harmony__", "__note_value__"):
        setattr(MV2H_global, attr, getattr(MV2H_global, attr) / len(y_true))

    return {
        "mv2h-multi-pitch": MV2H_global.__multi_pitch__,
        "mv2h-voice": MV2H_global.__voice__,
        "mv2h-meter": MV2H_global.__meter__,
        "mv2h-harmony": MV2H_global.__harmony__,
        "mv2h-note_value": MV2H_global.__note_value__,
        "mv2h-global": MV2H_global.mv2h,
    }


########################################################################## MUNSTER:
def compute_muster_metrics(y_true, y_pred):
    krn_parser = krnParser()
    muster_eval_script = "/MUSTER/evaluate_XML_voicePlus.sh"

    def krn2xml(in_file, tmpdir):
        xml_path = os.path.join(tmpdir, f"{uuid.uuid4()}.xml")
        converterm21.parse(in_file).write("musicxml", fp=xml_path)
        return xml_path

    def parse_output(out_file):
        with open(out_file, "r") as f:
            line = f.readline().strip()

        if ":" in line:
            names_part, values_part = line.split(":", 1)
            metric_names = [x.strip() for x in names_part.split(",")]
            metrics_values = values_part.strip().split()
        else:
            metric_names = [
                "PitchER", "MissRate", "ExtraRate", "OnsetER", "OffsetER",
                "MeanER", "VoiceER", "NewMeanER", "Pvoi", "Rvoi", "Fvoi",
                "ScaleErr", "HandER",
            ]
            metrics_values = line.split()
        metrics_dict = dict(zip(metric_names, metrics_values))
        return metrics_dict, metrics_dict["MeanER"]
    

    def eval_as_polyphonic(true_kern, pred_kern, tmpdir):
        true_xml = krn2xml(true_kern, tmpdir)
        pred_xml = krn2xml(pred_kern, tmpdir)
        out_file = os.path.join(tmpdir, f"{uuid.uuid4()}.txt")

        res = subprocess.run(
            [
                muster_eval_script,
                os.path.splitext(os.path.abspath(true_xml))[0],
                os.path.splitext(os.path.abspath(pred_xml))[0],
                os.path.splitext(os.path.abspath(out_file))[0]
            ],
            capture_output=True,
            text=True,
            check=True,
        )                
        output_dict, mean_er = parse_output(out_file)
        output_dict = {k: (float(v) if v.replace('.','',1).isdigit() else 100.0) for k, v in output_dict.items()}
        
        return output_dict, mean_er
    
    def eval_as_monophonic(true_kern, pred_kern, num_voices, tmpdir):
        per_voice_output = []
        for it_voice in range(num_voices):
            true_voice = divide_voice(true_kern, it_voice, tmpdir)
            pred_voice = divide_voice(pred_kern, it_voice, tmpdir)

            true_xml = krn2xml(true_voice, tmpdir)
            pred_xml = krn2xml(pred_voice, tmpdir)
            out_file = os.path.join(tmpdir, f"{uuid.uuid4()}.txt")

            res = subprocess.run(
                [
                    muster_eval_script,
                    os.path.splitext(os.path.abspath(true_xml))[0],
                    os.path.splitext(os.path.abspath(pred_xml))[0],
                    os.path.splitext(os.path.abspath(out_file))[0]
                ],
                capture_output=True,
                text=True,
                check=True,
            )                
            output_dict, mean_er = parse_output(out_file)
            output_dict = {k: (float(v) if v.replace('.','',1).isdigit() else 100.0) for k, v in output_dict.items()}
            per_voice_output.append(output_dict)

        # mean metrics across voices
        final_output_dict = {}
        for k in per_voice_output[0].keys():
            final_output_dict[k] = sum(d[k] for d in per_voice_output) / num_voices

        return final_output_dict, final_output_dict["MeanER"]

    muster_global = {
        "muster-pitch": 0.0,
        "muster-miss": 0.0,
        "muster-extra": 0.0,
        "muster-onset": 0.0,
        "muster-offset": 0.0,
        "muster-mean": 0.0,
        "muster-voice": 0.0,
        "muster-new-mean": 0.0,
    }
        
    for idx, (t, h) in enumerate(zip(y_true, y_pred)):
        with tempfile.TemporaryDirectory() as tmpdir:            
        
            true_kern = create_kern_file(t, krn_parser, get_number_of_voices(t), tmpdir)
            pred_kern = create_kern_file(h, krn_parser, get_number_of_voices(h), tmpdir)

            try:
                mean_ers, output_dicts = [], []
                for i in range(10):
                    output_dict, mean_er = eval_as_polyphonic(true_kern, pred_kern, tmpdir)
                    mean_ers.append(mean_er)
                    output_dicts.append(output_dict) 
                output_dict = output_dicts[mean_ers.index(min(mean_ers))]
            
                # if any of the metrics is nan, raise an error to trigger per-voice evaluation
                if any(v != v for v in output_dict.values()):
                    raise Exception("MUSTER evaluation resulted in NaN metric value.")

                muster_global["muster-pitch"] += float(output_dict["PitchER"])
                muster_global["muster-miss"] += float(output_dict["MissRate"])
                muster_global["muster-extra"] += float(output_dict["ExtraRate"])
                muster_global["muster-onset"] += float(output_dict["OnsetER"])
                muster_global["muster-offset"] += float(output_dict["OffsetER"])
                muster_global["muster-mean"] += float(output_dict["MeanER"])
                muster_global["muster-voice"] += float(output_dict["VoiceER"])
                muster_global["muster-new-mean"] += float(output_dict["NewMeanER"])


            except Exception as e:
                try:
                    mean_ers, output_dicts = [], []
                    for i in range(10):
                        output_dict = eval_as_monophonic(true_kern, pred_kern, get_number_of_voices(t), tmpdir)
                        mean_ers.append(output_dict["MeanER"])
                        output_dicts.append(output_dict)
                    output_dict = output_dicts[mean_ers.index(min(mean_ers))]

                    muster_global["muster-pitch"] += float(output_dict["PitchER"])
                    muster_global["muster-miss"] += float(output_dict["MissRate"])
                    muster_global["muster-extra"] += float(output_dict["ExtraRate"])
                    muster_global["muster-onset"] += float(output_dict["OnsetER"])
                    muster_global["muster-offset"] += float(output_dict["OffsetER"])
                    muster_global["muster-mean"] += float(output_dict["MeanER"])
                    muster_global["muster-voice"] += float(output_dict["VoiceER"])
                    muster_global["muster-new-mean"] += float(output_dict["NewMeanER"])
                except Exception as e:
                    muster_global["muster-pitch"] += 100.0
                    muster_global["muster-miss"] += 100.0
                    muster_global["muster-extra"] += 100.0
                    muster_global["muster-onset"] += 100.0
                    muster_global["muster-offset"] += 100.0
                    muster_global["muster-mean"] += 100.0
                    muster_global["muster-voice"] += 100.0
                    muster_global["muster-new-mean"] += 100.0


    if sum(v for v in muster_global.values()) > 0.0:
        for k in muster_global.keys():
            muster_global[k] /= len(y_true)
    else:
        muster_global = {k: 100.0 for k in muster_global}
        print("WARNING: All MUSTER evaluations failed, setting all MUSTER metrics to 100.0, please check the MUSTER evaluation script.")

    return muster_global
