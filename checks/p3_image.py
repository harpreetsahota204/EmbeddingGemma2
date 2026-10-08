"""P3 checks: image leaf through FiftyOne's DataLoader path (spawn workers)."""
import multiprocessing
import time

import numpy as np

import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.zoo as foz

NAME = "google/embeddinggemma-2"
CLONE = "eg2-p3-check"
CLASSES = ["airplane", "bird", "cat", "cow", "dog", "person", "car"]


def main():
    print("start method:", multiprocessing.get_start_method())
    if CLONE in fo.list_datasets():
        fo.delete_dataset(CLONE)
    ds = fo.load_dataset("quickstart").select_group_slices("original").clone(CLONE)
    print("samples:", len(ds), "| media_type:", ds.media_type)
    results = []

    def check(name, ok, detail=""):
        results.append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")

    try:
        model = foz.load_zoo_model(NAME)

        t0 = time.time()
        ds.compute_embeddings(model, embeddings_field="eg2", batch_size=16, num_workers=4)
        emb = np.stack(ds.values("eg2"))
        check("compute_embeddings, 4 spawn workers", emb.shape == (len(ds), 768),
              f"{emb.shape} in {time.time() - t0:.1f}s")
        check("unit norms", np.abs(np.linalg.norm(emb, axis=1) - 1).max() < 1e-5)

        ds.compute_embeddings(model, embeddings_field="eg2_default", batch_size=16)
        check("compute_embeddings, default num_workers",
              np.allclose(np.stack(ds.values("eg2_default")), emb, atol=1e-3))

        fob.compute_similarity(
            ds,
            embeddings="eg2",
            model=NAME,
            model_kwargs={"media_type": "image", "embedding_dim": 768},
            brain_key="eg2_sim",
        )
        fob.compute_visualization(ds, embeddings="eg2", method="pca", brain_key="eg2_viz")

        # reload index from the database, as the App does for text queries
        idx = ds.load_brain_results("eg2_sim")
        check("index reloads with prompt support", idx.config.supports_prompts)

        precisions = {}
        for c in ["airplane", "bird", "cat", "cow"]:
            view = ds.sort_by_similarity(f"a photo of a {c}", k=5, brain_key="eg2_sim")
            hits = [s.ground_truth is not None and c in [d.label for d in s.ground_truth.detections]
                    for s in view]
            precisions[c] = sum(hits) / 5
        p5 = np.mean(list(precisions.values()))
        check("text->image P@5 >= 0.9", p5 >= 0.9, precisions)

        model.classes = CLASSES
        ds.apply_model(model, label_field="eg2_zs", batch_size=16, num_workers=4)
        labels = ds.values("eg2_zs.label")
        check("zero-shot apply_model writes Classification", all(l in CLASSES for l in labels),
              {c: labels.count(c) for c in CLASSES})

        # images whose ground truth contains exactly one of CLASSES
        correct = total = 0
        for gt, pred in zip(ds.values("ground_truth.detections.label"), labels):
            present = set(gt or []) & set(CLASSES)
            if len(present) == 1:
                total += 1
                correct += pred in present
        check("zero-shot accuracy on single-class images", total > 0,
              f"{correct}/{total} = {correct / max(total, 1):.2f}")
    finally:
        fo.delete_dataset(CLONE)

    print(f"\n{sum(results)}/{len(results)} passed")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
