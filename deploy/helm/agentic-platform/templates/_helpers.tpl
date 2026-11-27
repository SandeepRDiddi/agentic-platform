{{- define "agentic.fullname" -}}
{{- .Release.Name | trunc 40 | trimSuffix "-" -}}
{{- end -}}

{{- define "agentic.labels" -}}
app.kubernetes.io/name: agentic-platform
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version }}
{{- end -}}

{{- define "agentic.selector" -}}
app.kubernetes.io/name: agentic-platform
app.kubernetes.io/instance: {{ .root.Release.Name }}
app.kubernetes.io/component: {{ .name }}
{{- end -}}

{{- define "agentic.image" -}}
{{ .Values.image.repository }}:{{ .Values.image.tag }}
{{- end -}}

{{/* Pod template body for a service. Call: include "agentic.pod" (dict "root" $ "name" $name "svc" $svc) | nindent 4 */}}
{{- define "agentic.pod" -}}
{{- $root := .root }}{{- $svc := .svc }}{{- $full := include "agentic.fullname" .root }}
metadata:
  labels:
    {{- include "agentic.selector" . | nindent 4 }}
    azure.workload.identity/use: "true"
  annotations:
    checksum/config: {{ include (print $root.Template.BasePath "/configmap.yaml") $root | sha256sum }}
spec:
  serviceAccountName: {{ $full }}
  automountServiceAccountToken: false
  {{- with $root.Values.imagePullSecrets }}
  imagePullSecrets: {{ toYaml . | nindent 4 }}
  {{- end }}
  terminationGracePeriodSeconds: {{ $svc.terminationGracePeriodSeconds | default 40 }}
  securityContext:
    runAsNonRoot: true
    runAsUser: 10001
    runAsGroup: 10001
    fsGroup: 10001
    seccompProfile: { type: RuntimeDefault }
  topologySpreadConstraints:
    - maxSkew: 1
      topologyKey: topology.kubernetes.io/zone
      whenUnsatisfiable: ScheduleAnyway
      labelSelector:
        matchLabels:
          {{- include "agentic.selector" . | nindent 10 }}
  containers:
    - name: app
      image: {{ include "agentic.image" $root | quote }}
      imagePullPolicy: {{ $root.Values.image.pullPolicy }}
      env:
        - name: SERVICE
          value: {{ .name | quote }}
        - name: SERVICE_NAME
          value: {{ .name | quote }}
        - name: SERVICE_VERSION
          value: {{ $root.Values.image.tag | quote }}
      envFrom:
        - configMapRef: { name: {{ $full }}-config }
        - secretRef: { name: {{ $root.Values.secretName }} }
      ports:
        - name: {{ $svc.portName | default "http" }}
          containerPort: {{ $svc.port | int }}
      {{- if eq ($svc.probes | default "http") "metrics" }}
      livenessProbe:
        httpGet: { path: /metrics, port: {{ $svc.portName | default "http" }} }
        initialDelaySeconds: 10
        periodSeconds: 20
        failureThreshold: 3
      {{- else }}
      startupProbe:
        httpGet: { path: /healthz, port: http }
        periodSeconds: 2
        failureThreshold: 60
      readinessProbe:
        httpGet: { path: /readyz, port: http }
        periodSeconds: 10
        failureThreshold: 3
      livenessProbe:
        httpGet: { path: /healthz, port: http }
        periodSeconds: 20
        failureThreshold: 3
      {{- end }}
      lifecycle:
        preStop:
          exec: { command: ["sleep", "5"] }   # let the LB drain before SIGTERM
      resources: {{ toYaml $svc.resources | nindent 8 }}
      securityContext:
        allowPrivilegeEscalation: false
        readOnlyRootFilesystem: true
        capabilities: { drop: ["ALL"] }
      volumeMounts:
        - { name: tmp, mountPath: /tmp }
  volumes:
    - { name: tmp, emptyDir: { sizeLimit: 64Mi } }
{{- end -}}
