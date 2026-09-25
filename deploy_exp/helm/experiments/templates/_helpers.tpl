{{/* Common labels: the release name is the sweep name. */}}
{{- define "experiments.labels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
qatfd.io/sweep: {{ .Release.Name }}
{{- end }}

{{/* Env vars shared by every container: where the cell manifest is, which cell we are, S3 + chroma layout. */}}
{{- define "experiments.commonEnv" -}}
- name: JOB_COMPLETION_INDEX
  valueFrom:
    fieldRef:
      fieldPath: metadata.annotations['batch.kubernetes.io/job-completion-index']
- name: CELLS_JSON
  value: /manifest/cells.json
- name: RELEASE_NAME
  value: {{ .Release.Name | quote }}
- name: AWS_REGION
  value: {{ .Values.s3.region | quote }}
- name: AWS_DEFAULT_REGION
  value: {{ .Values.s3.region | quote }}
- name: DATA_BUCKET
  value: {{ .Values.s3.dataBucket | quote }}
- name: RESULTS_BUCKET
  value: {{ .Values.s3.resultsBucket | quote }}
- name: RESULTS_PREFIX
  value: {{ .Values.s3.resultsPrefix | quote }}
- name: JOBS_PREFIX
  value: {{ .Values.s3.jobsPrefix | quote }}
- name: SKUNK_CHROMADB_DIR
  value: /data/chromadb
- name: SKUNK_CHROMA_SERVER_HOST
  value: "127.0.0.1"
- name: SKUNK_CHROMA_SERVER_PORT
  value: {{ .Values.chroma.port | quote }}
- name: QATFD_BENCHMARKS_DIR
  value: /data/benchmarks
- name: NODE_NAME
  valueFrom:
    fieldRef:
      fieldPath: spec.nodeName
- name: POD_NAME
  valueFrom:
    fieldRef:
      fieldPath: metadata.name
{{- end }}
