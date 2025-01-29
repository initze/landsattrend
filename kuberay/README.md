# Kuberay setup

Install operator on existing Kubernetes cluster using instructions [here](https://docs.ray.io/en/master/cluster/kubernetes/getting-started/raycluster-quick-start.html).

Deploy RayClusters using values.yaml file in this directory.

```bash
helm install raycluster kuberay/ray-cluster --version 1.0.0 --values values.yaml
```

Forward port to local machine to access Ray dashboard and be able to submit jobs.

```bash
kubectl port-forward --address 0.0.0.0 service/raycluster-kuberay-head-svc 8265:8265
```

To take down cluster (when changing values in values.yaml file for example), run:

```bash
helm uninstall raycluster
```