create table if not exists public.chat_events (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  workflow_id text not null,
  query text not null,
  response text,
  events jsonb not null default '[]'::jsonb,
  created_at timestamptz not null default now()
);

create table if not exists public.expert_reviews (
  id uuid primary key default gen_random_uuid(),
  diagnosis_id uuid not null references public.diagnoses(id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  diagnosis text not null,
  confidence numeric not null check (confidence >= 0 and confidence <= 1),
  reviewer_decision text not null,
  corrected_disease text,
  notes text,
  created_at timestamptz not null default now()
);

alter table public.chat_events enable row level security;
alter table public.expert_reviews enable row level security;
create policy "users manage own chat events" on public.chat_events for all using (auth.uid() = user_id) with check (auth.uid() = user_id);
create policy "users manage reviews for own diagnoses" on public.expert_reviews for all
  using (auth.uid() = user_id and exists (select 1 from public.diagnoses d where d.id = diagnosis_id and d.user_id = auth.uid()))
  with check (auth.uid() = user_id and exists (select 1 from public.diagnoses d where d.id = diagnosis_id and d.user_id = auth.uid()));

create index if not exists diagnoses_user_created_idx on public.diagnoses(user_id, created_at desc);
create index if not exists reports_user_created_idx on public.reports(user_id, created_at desc);
